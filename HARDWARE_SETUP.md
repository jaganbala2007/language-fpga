# Hardware Setup & FPGA Deployment Guide: PYNQ-Z2

## Target Board: TUL PYNQ-Z2 (Xilinx Zynq-7000 XC7Z020-1CLG400C)

This document provides step-by-step instructions to build the FPGA bitstream in Xilinx Vivado and deploy the Multilingual Receptionist onto the physical PYNQ-Z2 development board.

---

## 1. Prerequisites
- **Host PC:** Vivado Design Suite (2020.2 or newer, including Vitis HLS).
- **Target Board:** PYNQ-Z2 development board with PYNQ v3.0+ Linux SD card image.
- **Ethernet / USB:** Connected to local network with static IP (default: `192.168.2.99`).

---

## 2. FPGA Hardware Synthesis Workflow

### Step 1: Export Fixed-Point Weights
Run the weight export utility on the host machine to generate the C/HLS header:
```bash
python src/fixed_point_nn.py
```
This writes `fpga/hls/model_weights_fixed.h` and `fpga/weights/*.hex`.

### Step 2: Generate Vivado Project & Bitstream
1. Open Vivado in batch or GUI mode:
```bash
vivado -mode batch -source fpga/scripts/create_pynq_project.tcl
```
2. In Vivado, run Implementation and Generate Bitstream:
```tcl
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
```
3. Export the hardware artifacts:
- Bitstream file: `pynq_multilingual_receptionist.runs/impl_1/design_1_wrapper.bit`
- Hardware handoff file: `design_1.hwh`

Copy these files to `fpga/bitstream/mlp_classifier.bit` and `fpga/bitstream/mlp_classifier.hwh`.

---

## 3. PYNQ-Z2 Board Deployment

1. Copy the project folder to the PYNQ-Z2 board over SSH/SCP:
```bash
scp -r . xilinx@192.168.2.99:/home/xilinx/language_fpga
```
2. SSH into the board:
```bash
ssh xilinx@192.168.2.99
cd /home/xilinx/language_fpga
```
3. Install runtime requirements:
```bash
pip3 install -r requirements.txt
```
4. Run live receptionist in FPGA hardware mode:
```bash
python3 src/cli_receptionist.py --backend fpga --audio Recording.wav
```
Or start the web application:
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

## 4. Hardware Verification & Register Map

The AXI4-Lite slave interface responds at base address `0x43C00000`:
- `0x00`: Control register (Write `0x01` to start, Read bit 1 for `done`).
- `0x10`: Class prediction register (`0` = English, `1` = Hindi, `2` = Tamil).
- `0x14 - 0x1C`: 16-bit logits for English, Hindi, and Tamil.
- `0x20 - 0x84`: 26 input features in Q8.8 fixed-point representation.
