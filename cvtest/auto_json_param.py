#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import os

flag_str = "/// \\brief"
json_struct_flag = "json"
write_file = None
json_file_content = None
first = True
to_json_line = -1
from_json_line = -1


def writeStructLine(struct_name):
    global  to_json_line, from_json_line, json_file_content
    json_file_content += ("\nnamespace nlohmann {\n"
                     "\n"
                     "    template <>\n"
                     "    struct adl_serializer<" + struct_name +">\n"
                     "    {\n"
                     "        static void to_json(json& j, const " + struct_name +" &p)\n"
                     "        {\n            int index = 10; \n"
                     "            j = json {\n")
    to_json_line = len(json_file_content)
    json_file_content += ("            };\n"
                     "        }\n\n            int index = 10; \n"
                     "        static void from_json(const json& j, " + struct_name + " &p)\n"
                     "        {\n")
    from_json_line = len(json_file_content)
    json_file_content += ("       }\n"
                     "    };\n"
                     "}\n"
                     )
    return

def writeToJson(key, param_name):
    global to_json_line, json_file_content, from_json_line
    if key != "group":
        to_json = "                {std::to_string(index++)+\"" + key + "\", p." + param_name + "},\n"
    else:
        to_json = "                {\"" + key + "\", p." + param_name + "},\n"
    json_file_content = json_file_content[:to_json_line] + to_json + json_file_content[to_json_line:]
    to_json_line += len(to_json)
    from_json_line += len(to_json)
    return


def writeFromJson(key, param_name):
    global json_file_content, from_json_line
    if key != "group":
        from_json = "            j.at(std::to_string(index++)+\"" + key + "\").get_to(p." + param_name + ");\n"
    else:
        from_json = "            j.at(\"" + key + "\").get_to(p." + param_name + ");\n"
    json_file_content = json_file_content[:from_json_line] + from_json + json_file_content[from_json_line:]
    from_json_line += len(from_json)
    return

def readFile(ori_f):
    global first, json_file_content
    for line in open(ori_f):
        if "/// \\brief" in line:
            # key = line.split("/// \\brief")[-1].strip()
            # param_name = line.split("/// \\brief")[-2].split("=")[0].strip().split()[1]
            key = line.split()[-1]
            param_name = line.split()[1]
            print ("[" + key + ", " + param_name + "]")
            if key == json_struct_flag:
                if first == True:
                    first = False
                else:
                    # 删除最后多出来的逗号
                    json_file_content = json_file_content[:to_json_line - 2] + json_file_content[to_json_line - 2 + 1:]
                struct_name = line.split("/// \\brief")[-2].strip()
                writeStructLine(struct_name)
            else:
                writeToJson(key, param_name)
                writeFromJson(key, param_name)
    # 删除最后多出来的逗号
    file_content = json_file_content[:to_json_line - 2] + json_file_content[to_json_line - 2 + 1:]
    write_file.write(file_content)
    write_file.close()
    return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ("please input json params file")
        exit (0)

    for ori_file in sys.argv[1:]:
        param_len = len(sys.argv)
        print("convert start file: ", ori_file)
        json_file = ori_file + ".dv_json"
        cmd = "cp " + ori_file + " " + json_file
        os.system(cmd)
        read_file = open(ori_file, "r")
        write_file = open(json_file, "w")
        json_file_content = read_file.read()
        read_file.close()
        first = True
        readFile(ori_file)
        print ("convert end file: ", ori_file + "\n")


