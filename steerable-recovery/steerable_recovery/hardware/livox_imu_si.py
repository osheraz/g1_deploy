"""Convert Livox acceleration from g to SI before DLIO, including calibration.

The selected DLIO revision ignores its scale matrix when imu/calibration=true.
Do not rely on that matrix for unit conversion. Simulation IMU is already SI.
"""
import argparse
import copy

STANDARD_GRAVITY=9.80665


def imu_to_si(message,input_unit):
    if input_unit not in ('g','si'):raise ValueError('input_unit must be g or si')
    out=copy.deepcopy(message)
    scale=STANDARD_GRAVITY if input_unit=='g' else 1.
    for axis in ('x','y','z'):
        setattr(out.linear_acceleration,axis,getattr(out.linear_acceleration,axis)*scale)
    if out.linear_acceleration_covariance[0]>=0:
        out.linear_acceleration_covariance=[float(v)*scale**2 for v in out.linear_acceleration_covariance]
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-topic',default='/livox/imu');p.add_argument('--output-topic',default='/livox/imu_si')
    p.add_argument('--input-unit',choices=['g','si'],required=True)
    a=p.parse_args()
    if a.input_topic==a.output_topic:p.error('Input and output topics must differ')
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu
    rclpy.init();node=rclpy.create_node('hv2_livox_imu_si')
    pub=node.create_publisher(Imu,a.output_topic,qos_profile_sensor_data)
    node.create_subscription(Imu,a.input_topic,lambda msg:pub.publish(imu_to_si(msg,a.input_unit)),qos_profile_sensor_data)
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
