# WiFi Hotspot Cheatsheet

SSID: `osher` | Password: `12345678` | Host IP: `10.42.0.1` | Dongle: `wlx5ce931823ffc` | Profile: `Hotspot-1`

## Setup (one time)

```bash
nmcli device wifi hotspot ifname wlx5ce931823ffc con-name Hotspot-1 ssid osher password 12345678
nmcli connection modify Hotspot-1 ipv4.method shared   # NAT internet from ethernet
```

## After reboot

```bash
nmcli connection up Hotspot-1
```

Optional — autostart on boot:

```bash
nmcli connection modify Hotspot-1 connection.autoconnect yes connection.autoconnect-priority 10
```

## Check status

```bash
nmcli device status | grep wlx        # want: connected  Hotspot-1
iw dev wlx5ce931823ffc info           # want: type AP
```

## Troubleshooting

```bash
# stuck on "connecting (need authentication)" to MWireless → force back to AP:
nmcli connection up Hotspot-1

# dongle detected?
lsusb | grep Realtek

# driver loaded?
lsmod | grep 88XXau
```
