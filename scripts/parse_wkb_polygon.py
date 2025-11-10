import base64, struct, sys

b64 = open("../original_datasets/geom64.txt","rb").read().strip()
buf = base64.b64decode(b64)

# byte 0: endian flag (1 = little)
assert buf[0] == 1
geom_type = struct.unpack_from("<I", buf, 1)[0]
assert geom_type == 3, f"not a Polygon, type={geom_type}"

num_rings = struct.unpack_from("<I", buf, 5)[0]
off = 9
print("rings:", num_rings)
for r in range(num_rings):
    npts = struct.unpack_from("<I", buf, off)[0]; off += 4
    print(f"ring {r} points:", npts)
    pts = []
    for _ in range(npts):
        x, y = struct.unpack_from("<dd", buf, off); off += 16
        pts.append((x, y))
    print("first 3 pts:", pts[:3])
    print("last 3 pts:", pts[-3:])
    print("closed ring?", pts[0] == pts[-1])
