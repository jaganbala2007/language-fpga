// Top-Level Pipelined MLP Classifier Core (26 Inputs -> 16 Hidden -> 3 Outputs)
// FPGA-Based Multilingual AI Receptionist
// Target: Xilinx PYNQ-Z2 (Zynq XC7Z020-1CLG400C)
// Arithmetic: 16-bit Signed Fixed-Point (Q8.8 Format)

`timescale 1ns / 1ps

module mlp_top (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    
    // Feature inputs: Supports both parallel 416-bit packed bus or serial interface
    input  wire [415:0] features_flat,      // 26 features x 16 bits = 416 bits (Q8.8)
    input  wire signed [15:0] feature_in,   // Serial input stream for 26 features
    input  wire [4:0]  feature_addr,        // Address 0 to 25
    input  wire        feature_wr_en,       // Write enable for serial input feature buffer
    
    output reg         done,
    output reg         busy,
    output reg  [1:0]  predicted_class,     // 0: English, 1: Hindi, 2: Tamil
    output reg  signed [15:0] score_english,
    output reg  signed [15:0] score_hindi,
    output reg  signed [15:0] score_tamil
);

    // FSM States
    localparam STATE_IDLE    = 3'd0;
    localparam STATE_LOAD    = 3'd1;
    localparam STATE_SCALE   = 3'd2;
    localparam STATE_LAYER1  = 3'd3;
    localparam STATE_LAYER2  = 3'd4;
    localparam STATE_ARGMAX  = 3'd5;
    localparam STATE_DONE    = 3'd6;

    reg [2:0] state, next_state;
    reg [5:0] feat_idx;
    reg [4:0] neuron_idx;

    // Internal data buffers (Q8.8 format)
    reg signed [15:0] raw_features [0:25];
    reg signed [15:0] scaled_features [0:25];
    reg signed [15:0] hidden_activations [0:15];
    reg signed [15:0] output_logits [0:2];
    reg signed [31:0] accumulator;

    integer idx;

    // Feature Scaler Mean & Inverse Scale ROM
    reg signed [15:0] scaler_mean_rom [0:25];
    reg signed [15:0] scaler_inv_scale_rom [0:25];

    // Layer 1 Weight ROM [26 * 16 = 416] & Bias ROM [16]
    reg signed [15:0] w1_rom [0:415];
    reg signed [15:0] b1_rom [0:15];

    // Layer 2 Weight ROM [16 * 3 = 48] & Bias ROM [3]
    reg signed [15:0] w2_rom [0:47];
    reg signed [15:0] b2_rom [0:2];

    // Initialize Parameter ROMs with trained fixed-point values (Q8.8)
    initial begin
        // Scaler Means
        scaler_mean_rom[0]  = -16'sd32768; scaler_mean_rom[1]  =  16'sd23789; scaler_mean_rom[2]  =  16'sd2154;
        scaler_mean_rom[3]  =  16'sd4379;  scaler_mean_rom[4]  = -16'sd1253;  scaler_mean_rom[5]  = -16'sd1498;
        scaler_mean_rom[6]  = -16'sd2400;  scaler_mean_rom[7]  = -16'sd1530;  scaler_mean_rom[8]  = -16'sd1540;
        scaler_mean_rom[9]  = -16'sd970;   scaler_mean_rom[10] = -16'sd1308;  scaler_mean_rom[11] = -16'sd707;
        scaler_mean_rom[12] = -16'sd841;   scaler_mean_rom[13] =  16'sd24853; scaler_mean_rom[14] =  16'sd10674;
        scaler_mean_rom[15] =  16'sd7310;  scaler_mean_rom[16] =  16'sd6514;  scaler_mean_rom[17] =  16'sd4659;
        scaler_mean_rom[18] =  16'sd4090;  scaler_mean_rom[19] =  16'sd3603;  scaler_mean_rom[20] =  16'sd3171;
        scaler_mean_rom[21] =  16'sd2974;  scaler_mean_rom[22] =  16'sd2529;  scaler_mean_rom[23] =  16'sd2318;
        scaler_mean_rom[24] =  16'sd2228;  scaler_mean_rom[25] =  16'sd2105;

        // Scaler Inverse Scales
        scaler_inv_scale_rom[0]  = 16'sd3;   scaler_inv_scale_rom[1]  = 16'sd8;   scaler_inv_scale_rom[2]  = 16'sd13;
        scaler_inv_scale_rom[3]  = 16'sd18;  scaler_inv_scale_rom[4]  = 16'sd20;  scaler_inv_scale_rom[5]  = 16'sd22;
        scaler_inv_scale_rom[6]  = 16'sd27;  scaler_inv_scale_rom[7]  = 16'sd30;  scaler_inv_scale_rom[8]  = 16'sd38;
        scaler_inv_scale_rom[9]  = 16'sd37;  scaler_inv_scale_rom[10] = 16'sd43;  scaler_inv_scale_rom[11] = 16'sd40;
        scaler_inv_scale_rom[12] = 16'sd51;  scaler_inv_scale_rom[13] = 16'sd8;   scaler_inv_scale_rom[14] = 16'sd20;
        scaler_inv_scale_rom[15] = 16'sd32;  scaler_inv_scale_rom[16] = 16'sd43;  scaler_inv_scale_rom[17] = 16'sd49;
        scaler_inv_scale_rom[18] = 16'sd68;  scaler_inv_scale_rom[19] = 16'sd81;  scaler_inv_scale_rom[20] = 16'sd95;
        scaler_inv_scale_rom[21] = 16'sd99;  scaler_inv_scale_rom[22] = 16'sd114; scaler_inv_scale_rom[23] = 16'sd145;
        scaler_inv_scale_rom[24] = 16'sd153; scaler_inv_scale_rom[25] = 16'sd166;

        // Layer 1 Biases
        b1_rom[0]  =  16'sd8;    b1_rom[1]  = -16'sd136; b1_rom[2]  =  16'sd191; b1_rom[3]  =  16'sd30;
        b1_rom[4]  =  16'sd232;  b1_rom[5]  =  16'sd214; b1_rom[6]  = -16'sd8;   b1_rom[7]  =  16'sd36;
        b1_rom[8]  = -16'sd147;  b1_rom[9]  =  16'sd259; b1_rom[10] = -16'sd64;  b1_rom[11] =  16'sd199;
        b1_rom[12] =  16'sd111;  b1_rom[13] = -16'sd171; b1_rom[14] = -16'sd113; b1_rom[15] =  16'sd201;

        // Layer 2 Biases
        b2_rom[0]  = -16'sd29;   b2_rom[1]  =  16'sd165;  b2_rom[2]  = -16'sd19;

        // Layer 1 Weights (W1)
        w1_rom[0]   = -16'sd388; w1_rom[1]   =  16'sd379; w1_rom[2]   =  16'sd331; w1_rom[3]   =  16'sd79;
        w1_rom[4]   = -16'sd39;  w1_rom[5]   =  16'sd104; w1_rom[6]   = -16'sd452; w1_rom[7]   =  16'sd176;
        w1_rom[8]   = -16'sd70;  w1_rom[9]   =  16'sd24;  w1_rom[10]  = -16'sd380; w1_rom[11]  = -16'sd51;
        w1_rom[12]  = -16'sd33;  w1_rom[13]  = -16'sd128; w1_rom[14]  =  16'sd49;  w1_rom[15]  = -16'sd60;
        
        w1_rom[16]  =  16'sd91;  w1_rom[17]  = -16'sd40;  w1_rom[18]  = -16'sd181; w1_rom[19]  = -16'sd281;
        w1_rom[20]  =  16'sd363; w1_rom[21]  = -16'sd245; w1_rom[22]  = -16'sd80;  w1_rom[23]  = -16'sd188;
        w1_rom[24]  = -16'sd212; w1_rom[25]  = -16'sd25;  w1_rom[26]  = -16'sd157; w1_rom[27]  = -16'sd6;
        w1_rom[28]  =  16'sd88;  w1_rom[29]  =  16'sd51;  w1_rom[30]  = -16'sd228; w1_rom[31]  =  16'sd17;
        
        w1_rom[32]  = -16'sd257; w1_rom[33]  =  16'sd149; w1_rom[34]  =  16'sd20;  w1_rom[35]  = -16'sd134;
        w1_rom[36]  = -16'sd374; w1_rom[37]  = -16'sd78;  w1_rom[38]  =  16'sd28;  w1_rom[39]  =  16'sd1;
        w1_rom[40]  =  16'sd163; w1_rom[41]  = -16'sd250; w1_rom[42]  = -16'sd233; w1_rom[43]  =  16'sd80;
        w1_rom[44]  = -16'sd103; w1_rom[45]  =  16'sd0;   w1_rom[46]  = -16'sd135; w1_rom[47]  =  16'sd141;

        w1_rom[48]  =  16'sd73;  w1_rom[49]  = -16'sd210; w1_rom[50]  = -16'sd62;  w1_rom[51]  =  16'sd176;
        w1_rom[52]  =  16'sd206; w1_rom[53]  =  16'sd52;  w1_rom[54]  = -16'sd120; w1_rom[55]  = -16'sd24;
        w1_rom[56]  = -16'sd91;  w1_rom[57]  = -16'sd159; w1_rom[58]  = -16'sd289; w1_rom[59]  =  16'sd88;
        w1_rom[60]  = -16'sd264; w1_rom[61]  =  16'sd53;  w1_rom[62]  = -16'sd93;  w1_rom[63]  = -16'sd144;

        w1_rom[64]  = -16'sd146; w1_rom[65]  = -16'sd165; w1_rom[66]  =  16'sd30;  w1_rom[67]  =  16'sd293;
        w1_rom[68]  = -16'sd231; w1_rom[69]  = -16'sd33;  w1_rom[70]  =  16'sd4;   w1_rom[71]  = -16'sd58;
        w1_rom[72]  = -16'sd228; w1_rom[73]  =  16'sd3;   w1_rom[74]  =  16'sd2;   w1_rom[75]  =  16'sd24;
        w1_rom[76]  =  16'sd48;  w1_rom[77]  = -16'sd71;  w1_rom[78]  = -16'sd51;  w1_rom[79]  = -16'sd62;

        w1_rom[80]  = -16'sd87;  w1_rom[81]  = -16'sd140; w1_rom[82]  =  16'sd132; w1_rom[83]  = -16'sd75;
        w1_rom[84]  =  16'sd46;  w1_rom[85]  = -16'sd124; w1_rom[86]  =  16'sd164; w1_rom[87]  =  16'sd99;
        w1_rom[88]  = -16'sd61;  w1_rom[89]  = -16'sd52;  w1_rom[90]  = -16'sd46;  w1_rom[91]  =  16'sd115;
        w1_rom[92]  = -16'sd27;  w1_rom[93]  = -16'sd43;  w1_rom[94]  = -16'sd25;  w1_rom[95]  =  16'sd29;

        w1_rom[96]  =  16'sd144; w1_rom[97]  = -16'sd81;  w1_rom[98]  = -16'sd49;  w1_rom[99]  = -16'sd55;
        w1_rom[100] = -16'sd172; w1_rom[101] = -16'sd38;  w1_rom[102] = -16'sd28;  w1_rom[103] = -16'sd132;
        w1_rom[104] =  16'sd78;  w1_rom[105] =  16'sd25;  w1_rom[106] = -16'sd81;  w1_rom[107] =  16'sd35;
        w1_rom[108] = -16'sd12;  w1_rom[109] = -16'sd165; w1_rom[110] = -16'sd236; w1_rom[111] = -16'sd107;

        w1_rom[112] =  16'sd200; w1_rom[113] =  16'sd74;  w1_rom[114] = -16'sd51;  w1_rom[115] =  16'sd100;
        w1_rom[116] = -16'sd5;   w1_rom[117] = -16'sd142; w1_rom[118] =  16'sd106; w1_rom[119] =  16'sd117;
        w1_rom[120] =  16'sd140; w1_rom[121] =  16'sd30;  w1_rom[122] = -16'sd67;  w1_rom[123] = -16'sd143;
        w1_rom[124] = -16'sd190; w1_rom[125] = -16'sd79;  w1_rom[126] =  16'sd122; w1_rom[127] =  16'sd43;

        w1_rom[128] = -16'sd137; w1_rom[129] =  16'sd158; w1_rom[130] = -16'sd24;  w1_rom[131] =  16'sd42;
        w1_rom[132] = -16'sd111; w1_rom[133] = -16'sd112; w1_rom[134] =  16'sd158; w1_rom[135] =  16'sd5;
        w1_rom[136] =  16'sd100; w1_rom[137] =  16'sd196; w1_rom[138] = -16'sd80;  w1_rom[139] =  16'sd155;
        w1_rom[140] =  16'sd251; w1_rom[141] = -16'sd331; w1_rom[142] = -16'sd11;  w1_rom[143] = -16'sd177;

        w1_rom[144] =  16'sd37;  w1_rom[145] = -16'sd11;  w1_rom[146] = -16'sd132; w1_rom[147] =  16'sd140;
        w1_rom[148] = -16'sd134; w1_rom[149] = -16'sd118; w1_rom[150] = -16'sd10;  w1_rom[151] = -16'sd76;
        w1_rom[152] = -16'sd234; w1_rom[153] =  16'sd19;  w1_rom[154] =  16'sd36;  w1_rom[155] = -16'sd128;
        w1_rom[156] = -16'sd44;  w1_rom[157] = -16'sd52;  w1_rom[158] = -16'sd75;  w1_rom[159] =  16'sd229;

        w1_rom[160] = -16'sd18;  w1_rom[161] = -16'sd7;   w1_rom[162] = -16'sd139; w1_rom[163] =  16'sd46;
        w1_rom[164] = -16'sd25;  w1_rom[165] =  16'sd52;  w1_rom[166] = -16'sd70;  w1_rom[167] =  16'sd52;
        w1_rom[168] = -16'sd152; w1_rom[169] = -16'sd155; w1_rom[170] = -16'sd33;  w1_rom[171] = -16'sd4;
        w1_rom[172] =  16'sd72;  w1_rom[173] = -16'sd83;  w1_rom[174] =  16'sd12;  w1_rom[175] =  16'sd17;

        w1_rom[176] =  16'sd35;  w1_rom[177] =  16'sd58;  w1_rom[178] =  16'sd98;  w1_rom[179] = -16'sd59;
        w1_rom[180] = -16'sd70;  w1_rom[181] = -16'sd56;  w1_rom[182] =  16'sd87;  w1_rom[183] =  16'sd265;
        w1_rom[184] = -16'sd112; w1_rom[185] =  16'sd11;  w1_rom[186] = -16'sd114; w1_rom[187] = -16'sd127;
        w1_rom[188] =  16'sd87;  w1_rom[189] = -16'sd53;  w1_rom[190] = -16'sd241; w1_rom[191] =  16'sd9;

        w1_rom[192] =  16'sd167; w1_rom[193] =  16'sd6;   w1_rom[194] = -16'sd115; w1_rom[195] = -16'sd204;
        w1_rom[196] =  16'sd22;  w1_rom[197] =  16'sd42;  w1_rom[198] =  16'sd119; w1_rom[199] =  16'sd200;
        w1_rom[200] =  16'sd144; w1_rom[201] = -16'sd23;  w1_rom[202] =  16'sd55;  w1_rom[203] =  16'sd330;
        w1_rom[204] =  16'sd4;   w1_rom[205] = -16'sd106; w1_rom[206] = -16'sd70;  w1_rom[207] =  16'sd71;

        w1_rom[208] = -16'sd29;  w1_rom[209] = -16'sd85;  w1_rom[210] = -16'sd152; w1_rom[211] =  16'sd42;
        w1_rom[212] = -16'sd19;  w1_rom[213] =  16'sd71;  w1_rom[214] =  16'sd129; w1_rom[215] = -16'sd92;
        w1_rom[216] = -16'sd92;  w1_rom[217] =  16'sd65;  w1_rom[218] =  16'sd8;   w1_rom[219] =  16'sd79;
        w1_rom[220] =  16'sd24;  w1_rom[221] =  16'sd54;  w1_rom[222] = -16'sd219; w1_rom[223] = -16'sd151;

        w1_rom[224] =  16'sd206; w1_rom[225] =  16'sd67;  w1_rom[226] =  16'sd241; w1_rom[227] = -16'sd39;
        w1_rom[228] =  16'sd33;  w1_rom[229] = -16'sd18;  w1_rom[230] =  16'sd252; w1_rom[231] = -16'sd118;
        w1_rom[232] = -16'sd153; w1_rom[233] = -16'sd280; w1_rom[234] = -16'sd278; w1_rom[235] =  16'sd54;
        w1_rom[236] =  16'sd28;  w1_rom[237] = -16'sd272; w1_rom[238] =  16'sd84;  w1_rom[239] = -16'sd437;

        w1_rom[240] =  16'sd179; w1_rom[241] =  16'sd2;   w1_rom[242] =  16'sd55;  w1_rom[243] =  16'sd50;
        w1_rom[244] = -16'sd177; w1_rom[245] =  16'sd72;  w1_rom[246] =  16'sd28;  w1_rom[247] =  16'sd30;
        w1_rom[248] =  16'sd240; w1_rom[249] = -16'sd123; w1_rom[250] = -16'sd185; w1_rom[251] = -16'sd22;
        w1_rom[252] =  16'sd97;  w1_rom[253] =  16'sd122; w1_rom[254] = -16'sd188; w1_rom[255] = -16'sd259;

        w1_rom[256] =  16'sd72;  w1_rom[257] =  16'sd61;  w1_rom[258] = -16'sd16;  w1_rom[259] = -16'sd158;
        w1_rom[260] = -16'sd47;  w1_rom[261] = -16'sd65;  w1_rom[262] = -16'sd195; w1_rom[263] = -16'sd126;
        w1_rom[264] =  16'sd49;  w1_rom[265] = -16'sd206; w1_rom[266] =  16'sd71;  w1_rom[267] =  16'sd45;
        w1_rom[268] = -16'sd111; w1_rom[269] =  16'sd39;  w1_rom[270] =  16'sd22;  w1_rom[271] =  16'sd20;

        w1_rom[272] = -16'sd21;  w1_rom[273] =  16'sd171; w1_rom[274] = -16'sd75;  w1_rom[275] = -16'sd88;
        w1_rom[276] =  16'sd23;  w1_rom[277] =  16'sd142; w1_rom[278] =  16'sd109; w1_rom[279] =  16'sd61;
        w1_rom[280] =  16'sd17;  w1_rom[281] =  16'sd218; w1_rom[282] =  16'sd19;  w1_rom[283] = -16'sd223;
        w1_rom[284] = -16'sd14;  w1_rom[285] = -16'sd40;  w1_rom[286] = -16'sd26;  w1_rom[287] = -16'sd50;

        w1_rom[288] =  16'sd6;   w1_rom[289] =  16'sd156; w1_rom[290] = -16'sd178; w1_rom[291] =  16'sd66;
        w1_rom[292] = -16'sd25;  w1_rom[293] = -16'sd79;  w1_rom[294] = -16'sd154; w1_rom[295] = -16'sd52;
        w1_rom[296] =  16'sd72;  w1_rom[297] = -16'sd97;  w1_rom[298] =  16'sd194; w1_rom[299] =  16'sd88;
        w1_rom[300] = -16'sd128; w1_rom[301] = -16'sd39;  w1_rom[302] = -16'sd24;  w1_rom[303] = -16'sd85;

        w1_rom[304] =  16'sd52;  w1_rom[305] =  16'sd38;  w1_rom[306] =  16'sd17;  w1_rom[307] = -16'sd12;
        w1_rom[308] = -16'sd279; w1_rom[309] = -16'sd40;  w1_rom[310] = -16'sd69;  w1_rom[311] = -16'sd144;
        w1_rom[312] = -16'sd174; w1_rom[313] = -16'sd37;  w1_rom[314] =  16'sd80;  w1_rom[315] = -16'sd42;
        w1_rom[316] = -16'sd8;   w1_rom[317] = -16'sd106; w1_rom[318] = -16'sd19;  w1_rom[319] = -16'sd38;

        w1_rom[320] = -16'sd100; w1_rom[321] =  16'sd131; w1_rom[322] = -16'sd17;  w1_rom[323] = -16'sd63;
        w1_rom[324] =  16'sd142; w1_rom[325] =  16'sd141; w1_rom[326] =  16'sd6;   w1_rom[327] = -16'sd17;
        w1_rom[328] =  16'sd151; w1_rom[329] = -16'sd121; w1_rom[330] =  16'sd82;  w1_rom[331] = -16'sd55;
        w1_rom[332] = -16'sd53;  w1_rom[333] = -16'sd79;  w1_rom[334] = -16'sd51;  w1_rom[335] = -16'sd14;

        w1_rom[336] = -16'sd35;  w1_rom[337] = -16'sd3;   w1_rom[338] = -16'sd48;  w1_rom[339] =  16'sd64;
        w1_rom[340] = -16'sd219; w1_rom[341] =  16'sd84;  w1_rom[342] = -16'sd204; w1_rom[343] =  16'sd103;
        w1_rom[344] = -16'sd78;  w1_rom[345] = -16'sd28;  w1_rom[346] =  16'sd34;  w1_rom[347] =  16'sd157;
        w1_rom[348] = -16'sd180; w1_rom[349] = -16'sd65;  w1_rom[350] =  16'sd37;  w1_rom[351] =  16'sd3;

        w1_rom[352] = -16'sd120; w1_rom[353] =  16'sd185; w1_rom[354] = -16'sd75;  w1_rom[355] = -16'sd100;
        w1_rom[356] = -16'sd144; w1_rom[357] = -16'sd32;  w1_rom[358] =  16'sd38;  w1_rom[359] = -16'sd163;
        w1_rom[360] = -16'sd148; w1_rom[361] = -16'sd120; w1_rom[362] = -16'sd6;   w1_rom[363] = -16'sd103;
        w1_rom[364] =  16'sd161; w1_rom[365] =  16'sd106; w1_rom[366] =  16'sd3;   w1_rom[367] =  16'sd110;

        w1_rom[368] = -16'sd18;  w1_rom[369] = -16'sd51;  w1_rom[370] =  16'sd14;  w1_rom[371] =  16'sd8;
        w1_rom[372] = -16'sd47;  w1_rom[373] = -16'sd123; w1_rom[374] =  16'sd85;  w1_rom[375] = -16'sd70;
        w1_rom[376] = -16'sd36;  w1_rom[377] = -16'sd98;  w1_rom[378] =  16'sd91;  w1_rom[379] = -16'sd92;
        w1_rom[380] = -16'sd40;  w1_rom[381] =  16'sd75;  w1_rom[382] =  16'sd114; w1_rom[383] =  16'sd81;

        w1_rom[384] = -16'sd90;  w1_rom[385] =  16'sd62;  w1_rom[386] = -16'sd97;  w1_rom[387] = -16'sd2;
        w1_rom[388] =  16'sd36;  w1_rom[389] =  16'sd49;  w1_rom[390] =  16'sd9;   w1_rom[391] =  16'sd64;
        w1_rom[392] = -16'sd13;  w1_rom[393] = -16'sd23;  w1_rom[394] = -16'sd102; w1_rom[395] =  16'sd36;
        w1_rom[396] = -16'sd23;  w1_rom[397] =  16'sd32;  w1_rom[398] = -16'sd23;  w1_rom[399] =  16'sd94;

        w1_rom[400] = -16'sd230; w1_rom[401] =  16'sd58;  w1_rom[402] = -16'sd47;  w1_rom[403] =  16'sd129;
        w1_rom[404] =  16'sd18;  w1_rom[405] =  16'sd72;  w1_rom[406] = -16'sd17;  w1_rom[407] = -16'sd71;
        w1_rom[408] =  16'sd80;  w1_rom[409] = -16'sd177; w1_rom[410] =  16'sd1;   w1_rom[411] =  16'sd136;
        w1_rom[412] = -16'sd48;  w1_rom[413] =  16'sd38;  w1_rom[414] = -16'sd96;  w1_rom[415] =  16'sd129;

        // Layer 2 Weights (W2)
        w2_rom[0]  =  16'sd318; w2_rom[1]  =  16'sd44;  w2_rom[2]  = -16'sd588;
        w2_rom[3]  =  16'sd140; w2_rom[4]  = -16'sd510; w2_rom[5]  =  16'sd279;
        w2_rom[6]  =  16'sd337; w2_rom[7]  = -16'sd130; w2_rom[8]  = -16'sd473;
        w2_rom[9]  =  16'sd222; w2_rom[10] =  16'sd105; w2_rom[11] = -16'sd472;
        w2_rom[12] = -16'sd215; w2_rom[13] =  16'sd159; w2_rom[14] =  16'sd320;
        w2_rom[15] =  16'sd74;  w2_rom[16] =  16'sd134; w2_rom[17] = -16'sd286;
        w2_rom[18] =  16'sd365; w2_rom[19] = -16'sd353; w2_rom[20] =  16'sd11;
        w2_rom[21] =  16'sd296; w2_rom[22] = -16'sd438; w2_rom[23] =  16'sd135;
        w2_rom[24] = -16'sd230; w2_rom[25] =  16'sd313; w2_rom[26] = -16'sd416;
        w2_rom[27] = -16'sd359; w2_rom[28] = -16'sd190; w2_rom[29] =  16'sd301;
        w2_rom[30] = -16'sd26;  w2_rom[31] = -16'sd378; w2_rom[32] =  16'sd482;
        w2_rom[33] = -16'sd333; w2_rom[34] = -16'sd37;  w2_rom[35] =  16'sd336;
        w2_rom[36] = -16'sd228; w2_rom[37] =  16'sd390; w2_rom[38] = -16'sd272;
        w2_rom[39] =  16'sd76;  w2_rom[40] = -16'sd394; w2_rom[41] =  16'sd462;
        w2_rom[42] =  16'sd16;  w2_rom[43] =  16'sd494; w2_rom[44] = -16'sd467;
        w2_rom[45] = -16'sd295; w2_rom[46] =  16'sd456; w2_rom[47] = -16'sd109;
    end

    // Input feature buffering (Supports both direct parallel load and serial writes)
    always @(posedge clk) begin
        if (feature_wr_en && feature_addr < 26) begin
            raw_features[feature_addr] <= feature_in;
        end
    end

    // Sequential Arithmetic State Machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            feat_idx <= 6'd0;
            neuron_idx <= 5'd0;
            accumulator <= 32'sd0;
            done <= 1'b0;
            busy <= 1'b0;
            predicted_class <= 2'd0;
            score_english <= 16'sd0;
            score_hindi   <= 16'sd0;
            score_tamil   <= 16'sd0;
            for (idx = 0; idx < 16; idx = idx + 1) hidden_activations[idx] <= 16'sd0;
            for (idx = 0; idx < 3;  idx = idx + 1) output_logits[idx] <= 16'sd0;
        end else begin
            state <= next_state;

            case (state)
                STATE_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        feat_idx <= 6'd0;
                        neuron_idx <= 5'd0;
                    end else begin
                        busy <= 1'b0;
                    end
                end

                STATE_LOAD: begin
                    // Unpack parallel flat features into internal array
                    for (idx = 0; idx < 26; idx = idx + 1) begin
                        raw_features[idx] <= features_flat[idx*16 +: 16];
                    end
                    feat_idx <= 6'd0;
                end

                STATE_SCALE: begin
                    // Standardization: z = ((raw - mean) * inv_scale) >>> 8
                    reg signed [31:0] diff;
                    reg signed [31:0] scaled_prod;
                    
                    diff = $signed({{16{raw_features[feat_idx][15]}}, raw_features[feat_idx]}) - 
                           $signed({{16{scaler_mean_rom[feat_idx][15]}}, scaler_mean_rom[feat_idx]});
                    scaled_prod = (diff * $signed({{16{scaler_inv_scale_rom[feat_idx][15]}}, scaler_inv_scale_rom[feat_idx]})) >>> 8;

                    // Saturate to 16-bit signed
                    if (scaled_prod > 32'sd32767)
                        scaled_features[feat_idx] <= 16'sd32767;
                    else if (scaled_prod < -32'sd32768)
                        scaled_features[feat_idx] <= -16'sd32768;
                    else
                        scaled_features[feat_idx] <= scaled_prod[15:0];

                    if (feat_idx == 6'd25) begin
                        feat_idx <= 6'd0;
                        neuron_idx <= 5'd0;
                        accumulator <= $signed({{16{b1_rom[0][15]}}, b1_rom[0]}) <<< 8;
                    end else begin
                        feat_idx <= feat_idx + 1'b1;
                    end
                end

                STATE_LAYER1: begin
                    // Accumulate dot product for current hidden neuron
                    accumulator <= accumulator + ($signed(scaled_features[feat_idx]) * $signed(w1_rom[feat_idx * 16 + neuron_idx]));

                    if (feat_idx == 6'd25) begin
                        // Finish neuron with ReLU activation
                        reg signed [31:0] acc_final;
                        reg signed [15:0] acc_scaled;
                        acc_final = accumulator + ($signed(scaled_features[feat_idx]) * $signed(w1_rom[feat_idx * 16 + neuron_idx]));
                        acc_scaled = acc_final >>> 8;
                        
                        // ReLU: max(0, x)
                        hidden_activations[neuron_idx] <= (acc_scaled > 16'sd0) ? acc_scaled : 16'sd0;

                        feat_idx <= 6'd0;
                        if (neuron_idx == 5'd15) begin
                            neuron_idx <= 5'd0;
                            accumulator <= $signed({{16{b2_rom[0][15]}}, b2_rom[0]}) <<< 8;
                        end else begin
                            neuron_idx <= neuron_idx + 1'b1;
                            accumulator <= $signed({{16{b1_rom[neuron_idx + 1][15]}}, b1_rom[neuron_idx + 1]}) <<< 8;
                        end
                    end else begin
                        feat_idx <= feat_idx + 1'b1;
                    end
                end

                STATE_LAYER2: begin
                    // Accumulate dot product for output classes
                    accumulator <= accumulator + ($signed(hidden_activations[feat_idx[3:0]]) * $signed(w2_rom[feat_idx[3:0] * 3 + neuron_idx[1:0]]));

                    if (feat_idx == 6'd15) begin
                        reg signed [31:0] acc_final;
                        acc_final = accumulator + ($signed(hidden_activations[feat_idx[3:0]]) * $signed(w2_rom[feat_idx[3:0] * 3 + neuron_idx[1:0]]));
                        output_logits[neuron_idx[1:0]] <= acc_final >>> 8;

                        feat_idx <= 6'd0;
                        if (neuron_idx == 5'd2) begin
                            neuron_idx <= 5'd0;
                        end else begin
                            neuron_idx <= neuron_idx + 1'b1;
                            accumulator <= $signed({{16{b2_rom[neuron_idx + 1][15]}}, b2_rom[neuron_idx + 1]}) <<< 8;
                        end
                    end else begin
                        feat_idx <= feat_idx + 1'b1;
                    end
                end

                STATE_ARGMAX: begin
                    score_english <= output_logits[0];
                    score_hindi   <= output_logits[1];
                    score_tamil   <= output_logits[2];

                    if (output_logits[1] > output_logits[0] && output_logits[1] >= output_logits[2]) begin
                        predicted_class <= 2'd1; // Hindi
                    end else if (output_logits[2] > output_logits[0] && output_logits[2] > output_logits[1]) begin
                        predicted_class <= 2'd2; // Tamil
                    end else begin
                        predicted_class <= 2'd0; // English
                    end
                end

                STATE_DONE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            endcase
        end
    end

    // FSM State Transition Graph
    always @(*) begin
        next_state = state;
        case (state)
            STATE_IDLE: begin
                if (start) next_state = STATE_LOAD;
            end
            STATE_LOAD: begin
                next_state = STATE_SCALE;
            end
            STATE_SCALE: begin
                if (feat_idx == 6'd25) next_state = STATE_LAYER1;
            end
            STATE_LAYER1: begin
                if (feat_idx == 6'd25 && neuron_idx == 5'd15) next_state = STATE_LAYER2;
            end
            STATE_LAYER2: begin
                if (feat_idx == 6'd15 && neuron_idx == 5'd2) next_state = STATE_ARGMAX;
            end
            STATE_ARGMAX: begin
                next_state = STATE_DONE;
            end
            STATE_DONE: begin
                if (!start) next_state = STATE_IDLE;
            end
        endcase
    end

endmodule
