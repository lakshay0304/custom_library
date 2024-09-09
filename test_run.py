

class TestRunManager:
    def __init__(self):
        self.test_run = None
        self.temp_file_name = TEMP_FILE_NAME
        self.save_to_disk = False
        self.disable_request = False

    def reset(self):
        self.test_run = None
        self.temp_file_name = TEMP_FILE_NAME
        self.save_to_disk = False
        self.disable_request = False
        
global_test_run_manager = TestRunManager()