# Vivado Project Generation Script for PYNQ-Z2 Multilingual MLP Classifier
# Target Board: PYNQ-Z2 (xc7z020clg400-1)
# Vivado Version: 2020.2 - 2024.2+

set project_name "pynq_multilingual_receptionist"
set project_dir  "./vivado_project"
set part_number  "xc7z020clg400-1"

puts "=========================================================="
puts "  CREATING VIVADO PROJECT FOR PYNQ-Z2 FPGA INFERENCE      "
puts "=========================================================="

# Create project
create_project $project_name $project_dir -part $part_number -force

# Set board properties if board files are installed
catch { set_property board_part tul.com.tw:pynq-z2:part0:1.0 [current_project] }

# Add RTL Source Files
add_files -norecurse [glob ./fpga/rtl/*.v]
update_compile_order -fileset sources_1

# Create Block Design
create_bd_design "design_1"

# Instantiate Zynq PS7 IP
set ps7 [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" } $ps7

# Instantiate AXI Interconnect
set axi_ic [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0]
set_property -dict [list CONFIG.NUM_MI {1} CONFIG.NUM_SI {1}] $axi_ic

# Instantiate Neural Network Accelerator (from RTL/HLS IP)
set mlp_core [create_bd_cell -type module -reference axi_lite_nn_controller mlp_accelerator]

# Connect Clocks and Resets
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins ps7/M_AXI_GP0_ACLK] [get_bd_pins axi_ic/ACLK] [get_bd_pins axi_ic/S00_ACLK] [get_bd_pins axi_ic/M00_ACLK] [get_bd_pins mlp_core/S_AXI_ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins axi_ic/ARESETN] [get_bd_pins axi_ic/S00_ARESETN] [get_bd_pins axi_ic/M00_ARESETN] [get_bd_pins mlp_core/S_AXI_ARESETN]

# Connect AXI Master from PS to Interconnect
connect_bd_intf_net [get_bd_intf_pins ps7/M_AXI_GP0] [get_bd_intf_pins axi_ic/S00_AXI]

# Connect Interconnect to MLP Core
connect_bd_intf_net [get_bd_intf_pins axi_ic/M00_AXI] [get_bd_intf_pins mlp_core/S_AXI]

# Auto-assign Base Address: 0x43C00000
assign_bd_address [get_bd_addr_segs {mlp_accelerator/S_AXI/reg0 }]
set_property range 64K [get_bd_addr_segs {ps7/Data/SEG_mlp_accelerator_reg0}]
set_property offset 0x43C00000 [get_bd_addr_segs {ps7/Data/SEG_mlp_accelerator_reg0}]

# Validate Block Design
validate_bd_design
save_bd_design

# Create HDL Wrapper
set wrapper_file [make_wrapper -files [get_files design_1.bd] -top]
add_files -norecurse $wrapper_file
update_compile_order -fileset sources_1

puts "[✓] Block Design generated successfully with AXI Base Address: 0x43C00000"
puts "[*] To generate bitstream, run: launch_runs impl_1 -to_step write_bitstream"
