import time
import subprocess
import shlex
import cv2


def check_rtsp_rtmp_stream(url, timeout=4, is_rtmp=False):
    try:
        transport = 'tcp' if is_rtmp else 'udp'
        cmd = f'ffmpeg -loglevel quiet -rtsp_transport tcp -i {url} -t 1 -f null -'
        args = shlex.split(cmd)
        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=True)
        return True, "RTSP流可用"
    except subprocess.TimeoutExpired:
        return False, "RTSP流连接超时"
    except subprocess.CalledProcessError as e:
        error_message = result.stderr.decode('utf-8')
        return False, f"RTSP流错误：{error_message}"


if __name__ == '__main__':
    s = time.time()
    rtsp_list = ["rtsp://192.168.130.183:554", "rtsp://192.168.130.182:554", "rtsp://192.168.130.182:554", "rtsp://192.168.130.188:554"]
    for i in rtsp_list:
        success, meg = check_rtsp_rtmp_stream(i)
        print(i, success, meg)
    print(time.time()-s)