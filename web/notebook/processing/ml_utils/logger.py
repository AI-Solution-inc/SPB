import csv
import logging

class Logger():
    def __init__(self, filename) -> None:
        self.filename = filename
        self.field_names = ["num_epoch",
                            "train_loss", "train_miou", "train_pix_acc",
                            "val_loss", "val_miou", "val_pix_acc"]

        with open(self.filename, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.field_names)
            writer.writeheader()

    def update_training_data(self, loss, miou, pix_acc):
        self.train_loss = loss
        self.train_miou = miou
        self.train_pix_acc = pix_acc
    
    def update_val_data(self, loss, miou, pix_acc):
        self.val_loss = loss
        self.val_miou = miou
        self.val_pix_acc = pix_acc

    def write_line(self, epoch_num):
        with open(self.filename, 'a') as file:
            writer = csv.writer(file, delimiter=',')
            row = [epoch_num,
                   self.train_loss, self.train_miou, self.train_pix_acc,
                   self.val_loss, self.val_miou, self.val_pix_acc]
            writer.writerow(row)



def create_base_logger(training_dir):
    # создание базового логгера  дообучения
    base_logger = logging.getLogger("Additional Training")
    base_logger.setLevel(logging.DEBUG)
    
    # очистка файла от предыдущих логов
    filename = f"{training_dir}/log.log"
    with open(filename, 'w'):
        pass

    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(filename)s: %(message)s')
    file_handler = logging.FileHandler(filename=filename)
    file_handler.setFormatter(formatter)
    base_logger.addHandler(file_handler)
    return base_logger