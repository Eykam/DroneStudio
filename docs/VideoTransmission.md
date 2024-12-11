# H.264 UDP Transmission Throughput Requirements

## Raspberry Pi Zero 2 W Throughput Capability

The **Raspberry Pi Zero 2 W** has the following network capabilities:

- **Wireless LAN:** 
  - **Standard:** 2.4 GHz 802.11n
  - **Theoretical Maximum Throughput:** 150 Mbps
  - **Practical Throughput:** Approximately **70-100 Mbps** due to environmental factors, interference, and protocol overhead.

## Estimating UDP Transmission Speed

To determine the **UDP transmission speed** required for various streaming configurations, follow these steps:

1. **Frame Distribution:**
   - **I-Frames per Second (I_fps):** 
        $$\ I_{\text{fps}} = \frac{\text{Frame Rate (fps)}}{\text{I-frame Interval}}$$
   - **P-Frames per Second (P_fps):** 
        $$\ P_{\text{fps}} = \text{Frame Rate} - I_{\text{fps}}$$

2. **Data Rate Calculation:**
   - **I-Frame Size (S_I):** Size of each I-frame in Bytes.
   - **P-Frame Size (S_P):** Size of each P-frame in Bytes (range: lower to upper bound).
   - **Total Data per Second (D):**  
        $$\ D = (I_{\text{fps}} \times S_I) + (P_{\text{fps}} \times S_P)$$

3. **Convert to Megabits per Second (Mbps):**
    $$\ \text{Data Rate (Mbps)} = \frac{D \times 8}{1,000}$$

4. **Account for Network Overhead (1.3x):**
    $$\ \text{Throughput with Overhead} = \text{Data Rate (Mbps)} \times 1.3$$

5. **Incorporate Forward Error Correction (FEC) Overhead (20% if enabled):**
   - **Without FEC:**  
        $$\ \text{Total Throughput} = \text{Throughput with Overhead}$$
   - **With FEC:**  
        $$\ \text{Total Throughput} = \text{Throughput with Overhead} \times 1.2$$

6. **Determine Raspberry Pi Zero 2 W Support:**
   - **Supported:** If **Total Throughput ≤ 100 Mbps**
   - **Not Supported:** If **Total Throughput > 100 Mbps**

---

## Frame Size Estimates for H.264 Level 4.2

| **Resolution** | **Frame Type**  | **Lower Bound Frame Size (Bytes)** | **Upper Bound Frame Size (Bytes)** |
|----------------|-----------------|------------------------------------|------------------------------------|
| **360p**       | **I-Frame**     | 30 KB                              | 102 KB                             |
|                | **P-Frame**     | 5 KB                               | 51 KB                              |
| **480p**       | **I-Frame**     | 51 KB                              | 153 KB                             |
|                | **P-Frame**     | 5 KB                               | 51 KB                              |
| **720p**       | **I-Frame**     | 102 KB                             | 307 KB                             |
|                | **P-Frame**     | 5 KB                               | 51 KB                              |
---

## Throughput Requirements Table

| **Resolution** | **Frame Rate (fps)** | **I-frame Interval (frames)** | **FEC** | **Throughput Lower Bound (Mbps)** | **Throughput Upper Bound (Mbps)** | **Raspberry Pi Zero 2 W Support**  |
|----------------|----------------------|-------------------------------|---------|-----------------------------------|-----------------------------------|------------------------------------|
| **360p**       | 105                  | 1                             | No      | 218.40                            | 218.40                            | ❌                                 |
|                | 105                  | 1                             | Yes     | 262.08                            | 262.08                            | ❌                                 |
| **480p**       | 90                   | 1                             | No      | 187.20                            | 187.20                            | ❌                                 |
|                | 90                   | 1                             | Yes     | 224.64                            | 224.64                            | ❌                                 |
| **720p**       | 75                   | 1                             | No      | 156.00                            | 156.00                            | ❌                                 |
|                | 75                   | 1                             | Yes     | 187.20                            | 187.20                            | ❌                                 |
| **360p**       | 105                  | 30                            | No      | 12.56                             | 60.06                             | ✔️                                 |
|                | 105                  | 30                            | Yes     | 15.07                             | 72.07                             | ✔️                                 |
| **480p**       | 90                   | 30                            | No      | 10.76                             | 61.78                             | ✔️                                 |
|                | 90                   | 30                            | Yes     | 12.92                             | 72.07                             | ✔️                                 |
| **720p**       | 75                   | 30                            | No      | 8.97                              | 51.48                             | ✔️                                 |
|                | 75                   | 30                            | Yes     | 10.76                             | 61.78                             | ✔️                                 |
| **360p**       | 105                  | 60                            | No      | 9.00                              | 57.33                             | ✔️                                 |
|                | 105                  | 60                            | Yes     | 10.80                             | 68.80                             | ✔️                                 |
| **480p**       | 90                   | 60                            | No      | 7.72                              | 49.14                             | ✔️                                 |
|                | 90                   | 60                            | Yes     | 9.27                              | 58.97                             | ✔️                                 |
| **720p**       | 75                   | 60                            | No      | 6.44                              | 49.14                             | ✔️                                 |
|                | 75                   | 60                            | Yes     | 7.72                              | 58.97                             | ✔️                                 |
| **360p**       | 105                  | 120                           | No      | 7.23                              | 55.97                             | ✔️                                 |
|                | 105                  | 120                           | Yes     | 8.67                              | 67.16                             | ✔️                                 |
| **480p**       | 90                   | 120                           | No      | 6.20                              | 47.97                             | ✔️                                 |
|                | 90                   | 120                           | Yes     | 7.44                              | 57.56                             | ✔️                                 |
| **720p**       | 75                   | 120                           | No      | 5.17                              | 47.97                             | ✔️                                 |
|                | 75                   | 120                           | Yes     | 6.20                              | 57.56                             | ✔️                                 |
---

## Legend

- **✔️** Supported by Raspberry Pi Zero 2 W
- **❌** Not supported by Raspberry Pi Zero 2 W

---
