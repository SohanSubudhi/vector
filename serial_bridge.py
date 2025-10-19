import time, json, requests, sys, serial

# ==== CONFIG ====
PORT = sys.argv[1] if len(sys.argv) > 1 else "COM5"  # Change to your port, e.g. /dev/tty.usbmodem1101
BAUD = 115200
SNAPSHOT_URL = "http://127.0.0.1:8000/snapshot"
PERIOD = 0.25  # 4 updates/sec

def pack(accel, pit_prob, speed):
    return json.dumps({"a": float(accel), "p": float(pit_prob), "s": float(speed)}) + "\n"

def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print(f"[Bridge] Connected to {PORT} @ {BAUD} baud")
    while True:
        try:
            r = requests.get(SNAPSHOT_URL, timeout=1.0)
            if r.status_code == 200:
                d = r.json()
                a = d.get("decision", {}).get("accel", 0.0)
                p = d.get("decision", {}).get("pit_prob", 0.0)
                s = d.get("state", {}).get("current_speed", 0.0)
                ser.write(pack(a, p, s).encode("utf-8"))
        except Exception as e:
            ser.write(b'{"a":0,"p":0,"s":0}\n')
        time.sleep(PERIOD)

if __name__ == "__main__":
    main()
