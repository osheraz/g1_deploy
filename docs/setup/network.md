# Robot network

The robot subnet is `192.168.123.0/24`. On a dedicated robot Ethernet interface,
keep the internet default route on Wi-Fi. Substitute your interface/profile names:

```bash
nmcli device status
nmcli connection show
sudo nmcli connection add type ethernet ifname enp132s0 con-name G1 \
  ipv4.method manual ipv4.addresses 192.168.123.222/24 \
  ipv4.never-default yes ipv6.method link-local
sudo nmcli connection up G1
ip -br addr show enp132s0
ping -c 2 192.168.123.161
cat /sys/class/net/enp132s0/speed
```

For a shared internet/robot switch, add the robot address to the existing DHCP
profile instead of replacing it:

```bash
sudo nmcli connection modify Internet +ipv4.addresses 192.168.123.222/24
sudo nmcli connection up Internet
```

Raw depth needs a gigabit link (`1000` above). A working depth viewer does not
prove enough bandwidth for depth, LiDAR and recording together.

The configured `g1_29` SSH alias reaches the head computer over link-local IPv6:

```bash
ssh g1_29
# Direct address for the configured robot; the suffix is the workstation interface.
ssh -6 unitree@fe80::d150:4901:5a1a:cb5%enp132s0
```

Keep IPv6 enabled on both ends. If SSH fails, inspect `ssh -G g1_29` and the
interface address; robot IPv4 connectivity alone does not test the head computer.
Changing network profiles can interrupt remote VS Code sessions.
