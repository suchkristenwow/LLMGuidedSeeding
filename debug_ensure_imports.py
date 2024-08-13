from LLMGuidedSeeding_pkg.utils.rehearsal_utils import ensure_imports 

with open("rawCode.txt","r") as f: 
    code = f.read() 

with open("./configs/std_imports.txt","r") as f:
    std_imports = f.read()

print()
print("..........................this is the result from ensure imports: ")
print(ensure_imports(code,std_imports)) 