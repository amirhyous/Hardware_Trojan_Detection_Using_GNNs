#----------RS232----------:



# file_name = "RS232-T1000"
# trojan_gates = ["U293", "U294", "U295", "U296", "U297", "U298", "U299", "U300", "U301", "U302", "U303", "U304", "U305"]
# trojan_gates_full = [file_name + "_" + gate for gate in ["U293", "U294", "U295", "U296", "U297", "U298", "U299", "U300", "U301", "U302", "U303", "U304", "U305"]]

# file_name = 'RS232-T1100'
# trojan_gates = ["iDatasend_reg_2_", "U293", "U294", "U295", "U296", "U297",  "U298", "U299", "U300", "U301", "U302", "U305"]
# trojan_gates_full = [file_name + "_" + gate for gate in ["iDatasend_reg_2_", "U293", "U294", "U295", "U296", "U297", "U298", "U299", "U300", "U301", "U302", "U305"]]

# file_name = 'RS232-T1200'
# trojan_gates = ["iDatasend_reg_1", "iDatasend_reg_2", "iDatasend_reg_3", "iDatasend_reg_4", "U292", "U293", "U294", "U295", "U296", "U297", "U300", "U301", "U302", "U303"]
# trojan_gates_full = [file_name + "_" + gate for gate in ["iDatasend_reg_1", "iDatasend_reg_2", "iDatasend_reg_3", "iDatasend_reg_4", "U292", "U293", "U294", "U295", "U296", "U297", "U300", "U301", "U302", "U303"]]

# file_name = 'RS232-T1300'
# trojan_gates = ["U292", "U293", "U294", "U295", "U296", "U297", "U302", "U303", "U304"]
# trojan_gates_full = [file_name + "_" + gate for gate in ["U292", "U293", "U294", "U295", "U296", "U297", "U302", "U303", "U304"]]

# file_name = 'RS232-T1400'
# trojan_gates = ["iDatasend_reg", "U292", "U293", "U294", "U295", "U296", "U297", "U298", "U299", "U300", "U301", "U302", "U303"]
# trojan_gates_full = [file_name + "_" + gate for gate in ["iDatasend_reg", "U292", "U293", "U294", "U295", "U296", "U297", "U298", "U299", "U300", "U301", "U302", "U303"]]

# file_name = 'RS232-T1500'
# trojan_gates = ["iDatasend_reg_2_", "U293", "U294", "U295", "U296", "U297", "U298", "U299", "U300", "U301", "U302", "U303", "U304", "U305"]
# trojan_gates_full = [file_name + "_" + gate for gate in ["iDatasend_reg_2_", "U293", "U294", "U295", "U296", "U297", "U298", "U299", "U300", "U301", "U302", "U303", "U304", "U305"]]

# file_name = 'RS232-T1600'
# trojan_gates = ["iDatasend_reg_1", "iDatasend_reg_2", "U293", "U294", "U295", "U296", "U297", "U300", "U301", "U302", "U303", "U304"]
# trojan_gates_full = [file_name + "_" + gate for gate in ["iDatasend_reg_1", "iDatasend_reg_2", "U293", "U294", "U295", "U296", "U297", "U300", "U301", "U302", "U303", "U304"]]



#----------S----------:


# file_name = "s15850-T100"
# trojan_gates = [
#     "Tg1_Trojan1", "Tg1_Trojan2", "Tg1_Trojan3", "Tg1_Trojan4", "Tg1_Trojan1234",
#     "Tg1_Trojan5", "Tg1_Trojan6", "Tg1_Trojan7", "Tg1_Trojan8", "Tg1_Trojan5678",
#     "Tg1_Tj_Trigger", "Tg1_Trigger", "Tg2_Trojan1", "Tg2_Trojan2", "Tg2_Trojan3",
#     "Tg2_Trojan4", "Tg2_Trojan1234", "Tg2_Trojan5", "Tg2_Trojan6", "Tg2_Trojan7",
#     "Tg2_Trojan8", "Tg2_Trojan5678", "Tg2_Tj_Trigger", "Tg2_Trigger", "INVtest_se",
#     "Trojan_Trigger", "Trojan_Paylaod"
# ]
# trojan_gates_full = [file_name + "_" + gate for gate in trojan_gates]


# file_name = "s35932-T100"
# trojan_gates = [
#     "Trojan1", "Trojan2", "Trojan3", "Trojan4", "Trojan1234_NOT",
#     "Trojan5", "Trojan6", "Trojan7", "Trojan8", "Trojan5678_NOT",
#     "INV_test_se", "Trojan_Trigger", "TrojanScanEnable",
#     "Trojan_Payload1", "Trojan_Payload2"
# ]
# trojan_gates_full = [file_name + "_" + gate for gate in trojan_gates]


# file_name = "s35932-T200"
# trojan_gates = [
#     "Trojan1", "Trojan2", "Trojan3", "Trojan4", "Trojan1234_NOT",
#     "Trojan5", "Trojan6", "Trojan7", "Trojan8", "Trojan5678_NOT",
#     "INVtest_se", "Trojan_Trigger"
# ]
# trojan_gates_full = [file_name + "_" + gate for gate in trojan_gates]


# file_name = "s35932-T300"
# trojan_gates = [
#     "Trojan1", "Trojan2", "Trojan3", "Trojan4", "Trojan1234_NOT",
#     "Trojan5", "Trojan6", "Trojan7", "Trojan8", "Trojan5678_NOT",
#     "INVtest_se", "Trojan_Trigger",
#     "TjPayload1", "TjPayload2", "TjPayload3", "TjPayload4", "TjPayload5",
#     "TjPayload6", "TjPayload7", "TjPayload8", "TjPayload9", "TjPayload10",
#     "TjPayload11", "TjPayload12", "TjPayload13", "TjPayload14", "TjPayload15",
#     "TjPayload16", "TjPayload17", "TjPayload18", "TjPayload19", "TjPayload20",
#     "TjPayload21", "TjPayload22", "TjPayload23", "TjPayload24"
# ]
# trojan_gates_full = [file_name + "_" + gate for gate in trojan_gates]

#*********************************************************************************************************
# file_name = "s38417-T100"
# trojan_gates = [
#     "Trojan1", "Trojan2", "Trojan3", "Trojan4", "Trojan1234_NOT",
#     "Trojan5", "Trojan6", "Trojan7", "Trojan8", "Trojan5678_NOT",
#     "Trojan_CLK_NOT", "Trojan_Payload"
# ]
# trojan_gates_full = [file_name + "_" + gate for gate in trojan_gates]


# file_name = "s38417-T200"
# trojan_gates = [
#     "Trojan1", "Trojan2", "Trojan3", "Trojan4", "Trojan1234",
#     "Trojan5", "Trojan6", "Trojan7", "Trojan8", "Trojan5678",
#     "Trojan_Trigger", 
#     "Trojan_Payload_1", "Trojan_Payload_2", "Trojan_Payload_3", "Trojan_Payload_4"
# ]
# trojan_gates_full = [file_name + "_" + gate for gate in trojan_gates]


# file_name = "s38417-T300"
# trojan_gates = [
#     "Trojan1", "Trojan2", "Trojan3", "Trojan4", "Trojan1234",
#     "Trojan5", "Trojan6", "Trojan7", "Trojan8", "Trojan_CLK_NOT",
#     "Trojan_Payload1", "Trojan_Payload2", "Trojan_Payload3", "Trojan_Payload",
#     "TrojanEnableGATE", 
#     "Ring_Inv1", "Ring_Inv2", "Ring_Inv3", "Ring_Inv4", "Ring_Inv5", 
#     "Ring_Inv6", "Ring_Inv7", "Ring_Inv8", "Ring_Inv9", "Ring_Inv10", 
#     "Ring_Inv11", "Ring_Inv12", "Ring_Inv13", "Ring_Inv14", "Ring_Inv15", 
#     "Ring_Inv16", "Ring_Inv17", "Ring_Inv18", "Ring_Inv19", "Ring_Inv20", 
#     "Ring_Inv21", "Ring_Inv22", "Ring_Inv23", "Ring_Inv24", "Ring_Inv25", 
#     "Ring_Inv26", "Ring_Inv27", "Ring_Inv28"
# ]
# trojan_gates_full = [file_name + "_" + gate for gate in trojan_gates]


# file_name = "s38584-T100"
# trojan_gates = [
#     "Trojan1", "Trojan2", "Trojan3", "Trojan4", "Trojan1234_NOT",
#     "Trojan5", "Trojan_Trigger", "Trojan_Payload"
# ]
# trojan_gates_full = [file_name + "_" + gate for gate in trojan_gates]

