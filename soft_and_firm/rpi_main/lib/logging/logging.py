from datetime import datetime

class Logging:
    def __init__(self,path_to_log_file):
        self.log_file = open(path_to_log_file,"a")
        self.log_file.write(f"\n===== PROGRAMM STARTED ON {datetime.now()} =====\n")
        print("\033[95m ===== LSMC:SortWarrior PuckCollect 2026 ===== \033[0m")

    def log_action(self,text):
        self.log_file.write(f"{datetime.now()}: [*] {text}\n")
        print(f"\033[0m [*] {text} \033[0m")

    def log_success(self,text):
        self.log_file.write(f"{datetime.now()}: [SUCCESS] {text}\n")
        print(f"\033[92m [SUCCESS] {text} \033[0m")

    def log_failure(self,text):
        self.log_file.write(f"{datetime.now()}: [FAILURE] {text}\n")
        print(f"\033[91m [FAILURE] {text} \033[0m")
