import csv
import struct
import serial
from datetime import datetime
from pathlib import Path

def read_lidar(port='COM3', n_scans=20):
    DATA_DIR = Path('.')
    file = DATA_DIR / 'box_with_things.csv'
    scan = -1
    previous_angle = None
    
    with serial.Serial(port, 230400, timeout=1) as ser:
        ser.setDTR(False)
        ser.setRTS(False)
        
        with open(file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['scan', 'angle_deg', 'dist_mm', 'intensity'])
            
            print(f"Починаємо запис {n_scans} обертів...")
            
            while scan < n_scans:
                if ser.read(1) != b'\x54':
                    continue
                packet = ser.read(46)
                
                if len(packet) != 46 or packet[0] != 0x2C:
                    continue
                    
                start = struct.unpack_from('<H', packet, 3)[0] / 100
                end = struct.unpack_from('<H', packet, 41)[0] / 100
                
                if previous_angle is not None and start < previous_angle:
                    scan += 1
                    print(f"Записано оберт: {scan}/{n_scans}")
                previous_angle = start
                
                if scan < 0 or scan >= n_scans:
                    continue
                    
                step = ((end - start) % 360) / 11
                for i in range(12):
                    distance, intensity = struct.unpack_from('<HB', packet, 5 + 3 * i)
                    if distance > 0:
                        angle = (start + step * i) % 360
                        writer.writerow([scan, angle, distance, intensity])
                        
    print("Дані записано")
    return file

read_lidar(port='COM3', n_scans=20)