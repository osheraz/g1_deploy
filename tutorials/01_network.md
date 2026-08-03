# 1. Network — ethernet to the robot

The G1 lives on `192.168.123.0/24`. Your machine needs a static address on that subnet.

## Wiring

Plug into the robot's ethernet port, directly or through a switch (TL-SG105 or any dumb
gigabit switch).

A **passive RJ45 splitter will not work** — those split 8 wires into two 100Mbps runs and
need a matching splitter at the far end. You need a real switch.

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
ping -c2 192.168.123.164         # onboard PC
ip route | grep 192.168.123      # kernel scope link route
```

Pings failing: robot powered and booted? link light on the switch? `ip neigh` shows
`INCOMPLETE` for an address nothing answers at.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| internet dies when the robot is plugged in | a robot profile with a gateway grabbed the default route |
| can only have robot *or* internet, never both | two profiles competing for one device — use case B |
| ping works, `rt/lowstate` doesn't | robot is in a mode that suppresses DDS, not a network problem |

No wired internet at all? See [wifi_hotspot.md](wifi_hotspot.md).

---

Next: [2. Install](02_install.md)
