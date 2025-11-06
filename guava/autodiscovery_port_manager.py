"""
autodiscovery_port_manager.py

Zero-config port coordination using UDP broadcast with unicast responses.
No files, no Redis, no setup - just works across the network!
"""

import socket
import json
import time
import threading
from typing import Optional, List, Dict, Tuple


class AutoDiscoveryPortManager:
    """
    Network-wide port coordination with ZERO setup.
    
    How it works:
    1. Job broadcasts "I want ports" on UDP
    2. Existing jobs UNICAST reply "I'm using port X" directly to sender
    3. Job picks next available port
    4. Job broadcasts "I claimed port Y"
    5. Job keeps broadcasting heartbeat every 10s
    
    Dead jobs: If no heartbeat for 30s, port is reclaimed.
    """
    
    DISCOVERY_PORT = 29499  # One below typical training ports
    BROADCAST_IP = "255.255.255.255"  # LAN broadcast
    HEARTBEAT_INTERVAL = 10.0  # seconds
    JOB_TIMEOUT = 30.0  # seconds
    
    def __init__(self):
        self.hostname = socket.gethostname()
        self.local_ip = self._get_local_ip()
        
        # Local job registry (jobs on THIS machine)
        self.local_jobs: Dict[str, dict] = {}
        
        # Network job registry (all jobs we've heard from)
        self.network_jobs: Dict[str, dict] = {}
        
        # UDP socket for discovery
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # ✅ macOS CRITICAL: SO_REUSEPORT allows multiple processes to bind same UDP port
        if hasattr(socket, 'SO_REUSEPORT'):
            try:
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                print(f"✅ SO_REUSEPORT enabled on discovery socket")
            except OSError as e:
                print(f"⚠️ SO_REUSEPORT failed: {e}")
        
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        try:
            self.sock.bind(("", self.DISCOVERY_PORT))
            print(f"🔍 Port discovery bound to UDP port {self.DISCOVERY_PORT}")
        except OSError as e:
            print(f"❌ Failed to bind discovery socket on port {self.DISCOVERY_PORT}: {e}")
            raise
        
        # Background listener thread
        self._running = True
        self._listener_thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="port_discovery_listener"
        )
        self._listener_thread.start()
        
        print(f"🔍 Port discovery active on {self.local_ip}")
    
    def _get_local_ip(self) -> str:
        """Get this machine's LAN IP."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
    
    def _broadcast(self, message: dict):
        """Send UDP broadcast to all machines on LAN + localhost."""
        data = json.dumps(message).encode('utf-8')
        
        try:
            # ✅ Send to localhost FIRST (for same-machine detection)
            self.sock.sendto(data, ("127.0.0.1", self.DISCOVERY_PORT))
        except Exception as e:
            print(f"⚠️ Localhost send failed: {e}")
        
        try:
            # Send to network broadcast (for other machines)
            self.sock.sendto(data, (self.BROADCAST_IP, self.DISCOVERY_PORT))
        except Exception as e:
            print(f"⚠️ Broadcast failed: {e}")
    
    def _unicast(self, message: dict, target_ip: str):
        """Send UDP message directly to specific IP."""
        data = json.dumps(message).encode('utf-8')
        try:
            self.sock.sendto(data, (target_ip, self.DISCOVERY_PORT))
        except Exception as e:
            print(f"⚠️ Unicast to {target_ip} failed: {e}")
    
    def _listen_loop(self):
        """Background thread: listen for discovery messages."""
        self.sock.settimeout(1.0)  # Non-blocking with 1s timeout
        
        while self._running:
            try:
                data, addr = self.sock.recvfrom(4096)
                msg = json.loads(data.decode('utf-8'))
                self._handle_message(msg, addr)
            except socket.timeout:
                # Cleanup stale jobs
                self._cleanup_stale_jobs()
            except Exception:
                pass
    
    def _handle_message(self, msg: dict, addr: Tuple[str, int]):
        """Process incoming discovery message."""
        msg_type = msg.get("type")
        sender_ip = addr[0]  # ✅ Extract sender's IP address
        
        if msg_type == "QUERY":
            # ✅ Someone asking "who's using ports?"
            # Reply DIRECTLY to sender with UNICAST (not broadcast)
            print(f"   📨 Received QUERY from {sender_ip}")
            
            for job_id, info in self.local_jobs.items():
                response = {
                    "type": "ANNOUNCE",
                    "job_id": job_id,
                    "hostname": self.hostname,
                    "ip": self.local_ip,
                    "base_port": info["base_port"],
                    "timestamp": time.time(),
                }
                
                # ✅ UNICAST reply directly to querier
                self._unicast(response, sender_ip)
                print(f"   📤 Sent ANNOUNCE to {sender_ip} (job={job_id}, port={info['base_port']})")
        
        elif msg_type == "ANNOUNCE":
            # Someone announcing their port usage
            job_id = msg.get("job_id")
            hostname = msg.get("hostname")
            ip = msg.get("ip")
            base_port = msg.get("base_port")
            
            # Update network registry
            key = f"{hostname}:{job_id}"
            self.network_jobs[key] = {
                "job_id": job_id,
                "hostname": hostname,
                "ip": ip,
                "base_port": base_port,
                "last_seen": time.time(),
            }
            
            # Log if it's a new discovery
            if sender_ip not in ["127.0.0.1", self.local_ip]:
                print(f"   📥 Discovered remote job: {job_id} on {hostname} using port {base_port}")
    
    def _cleanup_stale_jobs(self):
        """Remove jobs that haven't sent heartbeat in 30s."""
        now = time.time()
        stale = []
        
        for key, info in self.network_jobs.items():
            if now - info["last_seen"] > self.JOB_TIMEOUT:
                stale.append(key)
        
        for key in stale:
            info = self.network_jobs[key]
            print(f"🧹 Removed stale job: {info['job_id']} on {info['hostname']}")
            del self.network_jobs[key]
    
    def _is_port_range_free(self, base_port: int, num_ports: int = 9) -> bool:
        """
        Check if ports are free on THIS machine (OS check).
        Ports 0-7 are TCP, port 8 is UDP.
        """
        for offset in range(num_ports):
            port = base_port + offset
            
            # ✅ Port 8 is UDP (telemetry), all others are TCP
            if offset == 8:
                sock_type = socket.SOCK_DGRAM  # UDP
            else:
                sock_type = socket.SOCK_STREAM  # TCP
            
            try:
                s = socket.socket(socket.AF_INET, sock_type)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                
                # macOS also needs SO_REUSEPORT for proper port reuse detection
                if hasattr(socket, 'SO_REUSEPORT'):
                    try:
                        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                    except OSError:
                        pass
                
                s.bind(("0.0.0.0", port))
                s.close()
            except OSError:
                # Port is in use
                return False
        
        return True
    
    def allocate_port_range(
        self,
        job_id: str,
        preferred_base: int = 29500,
        max_attempts: int = 100,
    ) -> Optional[int]:
        """
        Auto-discover and allocate ports across the network.
        
        Process:
        1. Broadcast QUERY to discover active jobs
        2. Wait 2s for UNICAST replies
        3. Pick first free port range on THIS machine
        4. Broadcast ANNOUNCE to claim it
        5. Start heartbeat thread
        
        Returns:
            Base port number, or None if all ports taken
        """
        
        print(f"🔍 Discovering active jobs on network...")
        print(f"   My hostname: {self.hostname}")
        print(f"   My IP: {self.local_ip}")
        
        # Step 1: Ask network "who's out there?"
        self._broadcast({
            "type": "QUERY",
            "hostname": self.hostname,
            "ip": self.local_ip,
        })
        
        # Step 2: Wait for UNICAST replies
        print(f"   ⏳ Waiting 2s for responses...")
        time.sleep(2.0)  # Give jobs 2 seconds to respond
        
        # Step 3: Find ports used on THIS MACHINE ONLY
        used_ports_here = set()
        for key, info in self.network_jobs.items():
            if info["hostname"] == self.hostname:
                # Job on same machine - avoid its ports
                base = info["base_port"]
                for offset in range(9):
                    used_ports_here.add(base + offset)
                print(f"   🔒 Avoiding ports from job '{info['job_id']}': {base}-{base+8}")
        
        print(f"   📊 Found {len(self.network_jobs)} active jobs on network")
        print(f"   🔒 Ports in use on {self.hostname}: {sorted(used_ports_here) if used_ports_here else 'none'}")
        
        # Step 4: Find first available range
        for attempt in range(max_attempts):
            candidate = preferred_base + (attempt * 10)
            
            # Check if any port in [candidate, candidate+8] is taken
            conflict = False
            for offset in range(9):
                if (candidate + offset) in used_ports_here:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Double-check with OS
            if not self._is_port_range_free(candidate):
                print(f"   ⚠️ Port range {candidate}-{candidate+8} in use (OS check)")
                continue
            
            # SUCCESS! Claim this range
            self.local_jobs[job_id] = {
                "base_port": candidate,
                "timestamp": time.time(),
            }
            
            # Announce to network
            self._broadcast({
                "type": "ANNOUNCE",
                "job_id": job_id,
                "hostname": self.hostname,
                "ip": self.local_ip,
                "base_port": candidate,
                "timestamp": time.time(),
            })
            
            # Start heartbeat thread
            heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                args=(job_id,),
                daemon=True,
                name=f"heartbeat_{job_id}"
            )
            heartbeat_thread.start()
            
            print(f"✅ Allocated ports for '{job_id}' on {self.hostname}")
            print(f"   Ports: {candidate} → {candidate+8}")
            print(f"   Control:      {candidate+0}")
            print(f"   Metrics:      {candidate+1}")
            print(f"   Gradients:    {candidate+2}")
            print(f"   Checkpoints:  {candidate+7}")
            print(f"   Telemetry:    {candidate+8}")
            
            return candidate
        
        print(f"❌ No available ports found after {max_attempts} attempts")
        return None
    
    def _heartbeat_loop(self, job_id: str):
        """Keep broadcasting that we're alive."""
        while self._running and job_id in self.local_jobs:
            info = self.local_jobs[job_id]
            self._broadcast({
                "type": "ANNOUNCE",
                "job_id": job_id,
                "hostname": self.hostname,
                "ip": self.local_ip,
                "base_port": info["base_port"],
                "timestamp": time.time(),
            })
            time.sleep(self.HEARTBEAT_INTERVAL)
    
    def release_port_range(self, job_id: str) -> bool:
        """Release ports (stops heartbeat, announces goodbye)."""
        if job_id not in self.local_jobs:
            return False
        
        info = self.local_jobs[job_id]
        
        # Announce we're leaving
        self._broadcast({
            "type": "GOODBYE",
            "job_id": job_id,
            "hostname": self.hostname,
            "base_port": info["base_port"],
        })
        
        del self.local_jobs[job_id]
        print(f"🔓 Released ports for '{job_id}'")
        return True
    
    def list_network_jobs(self) -> List[dict]:
        """Get all active jobs across network."""
        return list(self.network_jobs.values())
    
    def get_network_overview(self):
        """Print pretty network overview."""
        jobs = self.list_network_jobs()
        
        if not jobs:
            print("📭 No active jobs detected")
            return
        
        # Group by hostname
        by_host = {}
        for info in jobs:
            host = info["hostname"]
            if host not in by_host:
                by_host[host] = []
            by_host[host].append(info)
        
        print("🌐 Network Training Jobs:")
        print("=" * 70)
        for host, host_jobs in by_host.items():
            ip = host_jobs[0]["ip"]
            print(f"\n🖥️  {host} ({ip})")
            for info in host_jobs:
                base = info["base_port"]
                age = time.time() - info["last_seen"]
                print(f"   └─ {info['job_id']:30s} ports {base:5d}-{base+8:5d}  ({age:.1f}s ago)")
    
    def shutdown(self):
        """Clean shutdown."""
        self._running = False
        for job_id in list(self.local_jobs.keys()):
            self.release_port_range(job_id)
        self.sock.close()


# ============================================================================
# Easy integration function
# ============================================================================

def get_training_ports(job_name: Optional[str] = None) -> int:
    """
    Zero-config port allocation.
    
    Usage:
        from autodiscovery_port_manager import get_training_ports
        
        base_port = get_training_ports("my_experiment")
        orch = Orchestrator(master_port=base_port, ...)
    """
    
    if job_name is None:
        job_name = f"train_{time.strftime('%Y%m%d_%H%M%S')}"
    
    pm = AutoDiscoveryPortManager()
    base_port = pm.allocate_port_range(job_id=job_name)
    
    if base_port is None:
        raise RuntimeError("❌ Could not allocate ports!")
    
    return base_port