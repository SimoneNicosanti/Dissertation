import json

with open("../../Results/Exec_Profile/Client_yolo11x-seg_0.5_exec_profile.json", "r") as f:
    json_dict = json.load(f)

sum_1 = 0
sum_2 = 0
for key in json_dict.keys():
    sum_1 += json_dict[key][0]
    sum_2 += json_dict[key][1]

print(sum_1, sum_2)