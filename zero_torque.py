"""Continuously command zero torque on G1-29; no camera or policy required.

Run on Bluefin with the robot supported and other motor publishers stopped:
    python g1_deploy/zero_torque.py

Ctrl+C stops publication. This does not restore the built-in controller or
guarantee the robot's mode after its communication watchdog expires.
"""
import argparse
from pathlib import Path
import sys
import threading
import time


def zero_command(mode_machine):
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC

    command = unitree_hg_msg_dds__LowCmd_()
    command.mode_pr = 0
    command.mode_machine = int(mode_machine)
    for index, motor in enumerate(command.motor_cmd):
        motor.mode = 1 if index < 29 else 0
        motor.q = motor.dq = motor.tau = motor.kp = motor.kd = 0.0
    command.crc = CRC().Crc(command)
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface', default='enp132s0')
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sonic.common.unitree_mode import release_active_motion_mode
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
    )
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    print('Zero torque: support the robot and stop other motor controllers.', flush=True)
    ChannelFactoryInitialize(0, args.interface)
    ready = threading.Event()
    state = {}
    crc = CRC()

    def receive(message):
        if not ready.is_set() and int(message.crc) == int(crc.Crc(message)):
            state['mode_machine'] = int(message.mode_machine)
            ready.set()

    subscriber = ChannelSubscriber('rt/lowstate', LowState_)
    publisher = None
    try:
        subscriber.Init(receive)
        if not ready.wait(5.0):
            raise RuntimeError('No valid G1 LowState received; check power and Ethernet.')
        # Retain the client while publishing, as in the deployment handover.
        motion_switcher = release_active_motion_mode()
        command = zero_command(state['mode_machine'])
        publisher = ChannelPublisher('rt/lowcmd', LowCmd_)
        publisher.Init()
        print('Sending tau=0, kp=0, kd=0 to all 29 joints. Ctrl+C stops.', flush=True)
        while True:
            if not publisher.Write(command):
                raise RuntimeError('Zero-torque command publication failed.')
            time.sleep(0.002)
    except KeyboardInterrupt:
        print('\nStopped publication; built-in controller was not restored.', flush=True)
    finally:
        if publisher is not None:
            publisher.Close()
        subscriber.Close()


if __name__ == '__main__':
    main()
