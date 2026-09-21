# 1. Network — ethernet to the robot

The G1 lives on `192.168.123.0/24`. Your machine needs a static address on that subnet.

## Wiring

Plug into the robot's ethernet port, directly or through a switch (TL-SG105 or any dumb
gigabit switch).

A **passive RJ45 splitter will not work** — those split 8 wires into two 100Mbps runs and
need a matching splitter at the far end. You need a real switch.

### Direct vs switch — pick by what you need

| Topology | Internet | Robot link speed | Depth transport |
| --- | --- | --- | --- |
| Ethernet direct to robot, internet over WiFi | WiFi | **1000Mb/s** full duplex (measured 2026-09-20) | `--encoding raw --max-mbps 0` (148 Mb/s) |
| Ethernet to a switch shared with internet (case B) | wired | capped at **100Mb/s** in practice | `--encoding png --max-mbps 40` (31 Mb/s) |

Raw depth does not fit on 100Mb/s. See [depth latency](../../notes/validation/depth_latency_20260917.md)
and [deploy.md](../steerable-recovery/docs/deploy.md#terminal-1-depth).

### Switching between them

Both profiles already hold `192.168.123.222/24`.

**Leaving the lab: wired internet and remote SSH.** Stop the depth/viewer terminals,
connect Ethernet to the wall network, activate `Internet`, then turn WiFi off:

```bash
sudo nmcli con up Internet ifname enp132s0
sudo nmcli radio wifi off
```

Bluefin's wired address is `141.212.48.126`, also resolved by
`bluefin.eecs.umich.edu` (verified September 20). On the other computer:

```bash
ssh osher@bluefin.eecs.umich.edu
```

Keeping MWireless active creates a directly connected `35.3.0.0/16` WiFi route.
Replies from Bluefin's wired address to a Mac on that subnet then select WiFi,
which can break incoming SSH. The wired route metric of 100 does not override
that more specific route. Turning WiFi off while using wired internet removes
this conflicting return path; both the wired address and SSH service stay active.

**Next robot session.** Connect WiFi before unplugging the wall Ethernet:

```bash
sudo nmcli radio wifi on
sudo nmcli con up MWireless ifname wlp131s0f0
```

Move the Ethernet cable directly to the robot, then select the saved direct profile:

```bash
sudo nmcli con up enp132s0 ifname enp132s0
cat /sys/class/net/enp132s0/speed       # expect 1000 on the verified direct setup
ping -c 2 192.168.123.161
ssh g1_29                            # after the Jetson boots
```

Restart the depth server and viewer using the camera guide. The installed server
and Jetson IPv6 link-local fix persist; they do not need to be reapplied daily.
`enp132s0` has `autoconnect no`, while `Internet` and `G1` have `yes`; explicitly
select the desired profile after cabling or rebooting.

Direct + WiFi is the default for streaming work. Take the switch only when you need wired
internet on the same port and can accept the cap. Confirm what you actually negotiated:

```bash
cat /sys/class/net/enp132s0/speed    # 1000 direct, 100 behind the shared switch
```

Find your interface:

```bash
ip -br link          # e.g. enp132s0
```

## Case A — ethernet is only for the robot

Static address, no gateway:

```bash
nmcli con add type ethernet ifname enp132s0 con-name G1 \
  ipv4.method manual ipv4.addresses 192.168.123.222/24
nmcli con up G1
```

Get internet over WiFi in this case.

## Case B — one ethernet port for robot *and* internet

NetworkManager allows one profile per device, so a separate `G1` profile kicks your
internet profile off — you end up choosing one or the other. The fix is to put **both
addresses on a single profile**.

### Find or create the internet profile

```bash
nmcli con show                   # list profiles
nmcli dev status                 # which profile is active on your interface
```

If a DHCP profile for the wired port already exists (often `Wired connection 1`), use its
name below. Otherwise create one — an ordinary DHCP profile, nothing special:

```bash
nmcli con add type ethernet ifname enp132s0 con-name Internet ipv4.method auto
nmcli con up Internet
```

Confirm it got an address and a default route:

```bash
ip -br addr show enp132s0
ip route | grep default
```

### Add the robot subnet to it

Note the `+`, which appends instead of replacing:

```bash
nmcli con mod Internet +ipv4.addresses 192.168.123.222/24
nmcli con up Internet
```

Costs nothing — the kernel routes by longest prefix, so `192.168.123.x` goes out on the
link and everything else via your normal gateway. Deliberately **no gateway** is set for
the robot subnet, so it cannot steal your default route.

To undo, drop just that address:

```bash
nmcli con mod Internet -ipv4.addresses 192.168.123.222/24
nmcli con up Internet
```

## Check

```bash
ip -br addr show enp132s0        # both 192.168.123.222/24 and your normal IP
ping -c2 192.168.123.161         # robot
ip route | grep 192.168.123      # kernel scope link route
```

`192.168.123.164` does not exist on the G1-29 — its Jetson is IPv6-only. Only `.161` and
`.120` answer on IPv4, and neither runs sshd.

Pings failing: robot powered and booted? link light on the switch? `ip neigh` shows
`INCOMPLETE` for an address nothing answers at.

## New G1-29: SSH over IPv6 (verified 2026-09-11, re-verified 2026-09-20)

The Jetson has **no IPv4**, so case A/B address settings do not reach it. The link-local
address is the only way in, and that requires IPv6 enabled on Bluefin's own profile:

```bash
nmcli -f ipv6.method con show enp132s0     # must NOT be 'disabled'
nmcli con mod enp132s0 ipv6.method link-local
nmcli con up enp132s0
ip -br addr show enp132s0                  # expect a fe80::... alongside 192.168.123.222/24
```

With `ipv6.method disabled` there is no link-local source address, so ssh to a link-local
target cannot send at all — it fails identically to the robot being off.

From **Bluefin**, connect to the new robot's Jetson with:

```bash
ssh -6 unitree@fe80::d150:4901:5a1a:cb5%enp132s0
```

Use the Jetson's login password. Verify its host-key fingerprint on first connection;
do not bypass host-key checking. `enp132s0` is **Bluefin's** Ethernet interface, not
the Jetson's. This address identifies this robot, not every G1.

The local Jetson console showed `eth0` UP with `LOWER_UP`, IPv6 addresses but **no
IPv4 address**, and SSH running/listening on port 22. Therefore the assumed
`192.168.123.164` address was not configured. Bluefin successfully pinged the IPv6
address, received an OpenSSH banner on port 22, and subsequently authenticated as
`unitree` to inspect and stage sensor software. The Ethernet path works; missing
IPv4 was not evidence of broken hardware.

No Ethernet settings, SSH services or robot controls were changed by this check.
Keep Bluefin's existing combined robot/Internet profile. IPv4 can be configured
separately later; NoMachine is not required for this SSH connection.

If the address changes, run `ip -br addr` **on the Jetson's local console** and find
the `fe80::...` address on its connected Ethernet interface. Use that address without
`/64`, followed by `%` and the connecting workstation's Ethernet interface.

Faster than walking to the console: sniff the Jetson's boot traffic from Bluefin and read
the address off the wire. Its DAD and mDNS announce it (`ubuntu.local`, MAC
`3c:6d:66:16:62:df`).

```bash
sudo timeout 60 tcpdump -nni enp132s0 -e ip6
```

`ip -6 neigh` showing `FAILED` for the address means nothing on the link owns it: either the
Jetson is down, or its stable-privacy address was regenerated.

## Jetson disappears periodically on the direct link (fixed 2026-09-20)

The Jetson's `Wired connection 1` used `ipv4.method auto` and `ipv6.method auto`.
Without an automatic network configuration source on the robot-only link, its
NetworkManager log showed 45-second DHCP timeouts, connection activation failures,
and five-minute retry pauses. The controller kept answering while `g1_29` disappeared.
Bluefin's WiFi/robot routes were correct; the failing profile was on the Jetson.

The applied fix **on the Jetson** was:

```bash
sudo nmcli connection modify 'Wired connection 1' ipv6.method link-local
sudo nmcli connection up 'Wired connection 1'
```

IPv4 stays `auto` with `may-fail yes`, so an optional IPv4 DHCP lease is still allowed.
IPv6 link-local now satisfies activation without a router. The existing address
`fe80::d150:4901:5a1a:cb5` and `g1_29` alias were preserved. After the next DHCP timeout,
the profile stayed `100 (connected)` instead of disconnecting.

Original profile backup on the Jetson:
`/home/unitree/osher/native_recovery/Wired_connection_1.before_linklocal_20260920.nmconnection`.
The exact reversible setting change is `ipv6.method auto` → `link-local`.
See the [verification receipt](../../logs/deployment_validation/depth_gigabit_20260920_160232/REPORT.md).

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| internet dies when the robot is plugged in | a robot profile with a gateway grabbed the default route |
| can only have robot *or* internet, never both | two profiles competing for one device — use case B |
| ping works, `rt/lowstate` doesn't | robot is in a mode that suppresses DDS, not a network problem |
| `ssh -6` to the Jetson times out | `ipv6.method disabled` on Bluefin's profile, or the Jetson is not booted |
| streaming throughput halves | `cat /sys/class/net/enp132s0/speed` reads 100 — you are on the shared switch |

No wired internet at all? See [wifi_hotspot.md](wifi_hotspot.md).

---

Next: [2. Install](02_install.md)
