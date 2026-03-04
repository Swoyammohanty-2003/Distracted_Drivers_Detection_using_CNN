import winsound

class AlertSystem:
    def __init__(self, threshold):
        self.threshold = threshold
        self.frame_count = 0

    def check_alert(self, predicted_class):
        if predicted_class != 0:
            self.frame_count += 1
        else:
            self.frame_count = 0

        if self.frame_count >= self.threshold:
            winsound.Beep(1000, 500)
            self.frame_count = 0
            return True
        return False
