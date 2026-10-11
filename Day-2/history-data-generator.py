import csv
import random
from pathlib import Path

header = ['temperature', 'pressure', 'vibration', 'rpm', 'current', 'state']

current_dir = Path(__file__).resolve().parent
output_file = current_dir / "sensor_reading.csv"

with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(header)

    for i in range(100):
        t = round(random.uniform(50, 95), 2)
        p = round(random.uniform(1.0, 10.0), 2)
        v = round(random.uniform(0.1, 5.0), 2)
        c = round(random.uniform(5.0, 20.0), 2)
        rpm = random.randint(500, 3500)

        if t > 90 or rpm > 3000 or p > 8 or c > 18.0 or v > 4.0:
            state = 'critical'
        elif t < 80 or rpm < 1000 or p < 3.0 or c < 6.0 or v < 1.0:
            state = 'warning'
        else:
            state = 'normal'

        row = [t, p, v, rpm, c, state]
        writer.writerow(row)