import os 

commonObjs = []
config_dir = "/home/kristen/LLMGuidedSeeding/configs"
with open(os.path.join(config_dir,"9k.names"),"r") as f: 
    lines = [line.strip().lower() for line in f]
    commonObjs.extend(lines)

with open(os.path.join(config_dir,"coco_classes.txt"),"r") as f:
    lines = [line.strip() for line in f]
    for line in lines:  
        idx = line.index("u")
        object = line[idx + 1:]
        if "__" in object:
            continue 
        if "}" in object:
            i0 = object.index("}")
            object = object[:i0]
        if "," in object:
            i0 = object.index(",")
            object = object[:i0]
        if "'" in object: 
            object = object[1:-1]
        if object not in commonObjs:
            print("new coco object!")
            commonObjs.append(object) 

with open(os.path.join(config_dir,"commonObjs.txt"),"w") as f:
    for object in commonObjs:
        print("object: ",object)
        f.write(object + "\n") 

