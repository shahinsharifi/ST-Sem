import torch
from torch.autograd import Variable
import util.utils as utils
from util import dataset
from PIL import Image
import util.crnn as crnn
import os
from termcolor import colored

class SceneTextRecognizer:

    def __init__(self):
        self.model_path = './models/crnn/crnn.pth'
        self.img_path = './temp/crops/'
        self.alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
        self.model = crnn.CRNN(32, 1, 37, 256)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(self.model_path))


    def recognize(self, inputName):

        converter = utils.strLabelConverter(self.alphabet)

        transformer = dataset.resizeNormalize((100, 32))
        result = []

        for filename in os.listdir(self.img_path + inputName + '/'):
            if filename.endswith(".jpg"):
                image = Image.open(os.path.join(self.img_path + inputName, filename)).convert('L')
                image = transformer(image)
                if torch.cuda.is_available():
                    image = image.cuda()
                image = image.view(1, *image.size())
                image = Variable(image)

                self.model.eval()
                preds = self.model(image)

                _, preds = preds.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)

                preds_size = Variable(torch.IntTensor([preds.size(0)]))
                raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
                sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
                if(str(sim_pred) not in result):
                    result.append(str(sim_pred))
                os.remove(os.path.join(self.img_path + inputName, filename))
                #print('%-20s => %-20s' % (raw_pred, sim_pred))

            else:
                continue
        del self.model
        os.rmdir(self.img_path + inputName + '/')
        return result


    def releaseMemory(self):
        del self.model