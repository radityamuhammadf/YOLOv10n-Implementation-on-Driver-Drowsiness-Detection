import jtop
from jtop.jtop import jtop
import cv2
import time

def test_jtop():
    with jtop() as jetson:
        print("Jetson stats initialized successfully")
        # print(f"CPU Usage: {jetson.stats['CPU']['usage']}")
        print(f"GPU Usage: {jetson.stats['GPU']}")
        # print(f"CPU Usage: {jetson.stats['CPU']['val']} %") #CPU usage on highest usage?
        # Access the correct key for overall CPU usage
        stats = jetson.stats
        print(stats)
        cpu_usages = []
        for key in stats:
            if key.startswith('CPU') and key[3:].isdigit():
                usage = stats[key]
                if usage != 'OFF':
                    cpu_usages.append(usage)
        
        if cpu_usages:
            overall_cpu_usage = sum(cpu_usages) / len(cpu_usages)
            print(f"Overall CPU usage: {overall_cpu_usage:.2f}%")
        else:
            print("No CPU usage data available.")
            # print(f"CPU Usage: {jetson.processes} %")
        print(f"Memory Usage: {jetson.stats['RAM']}")
        # print(f"Memory Usage: {jetson.stats['MEM']['use']} MB")

if __name__ == "__main__":
    test_jtop()
