import os
import time
import warnings
import operator

import cv2
from termcolor import colored
from scene_recognizer import SceneRecognizer
from scene_text_detector import SceneTextDetector
from scene_text_recognizer import SceneTextRecognizer
from word_similarity import WordSimilarity
from spellchecker import SpellChecker



os.environ['GLOG_minloglevel'] = '2'
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():

    print 'Loading models ...'

    inputFolder1 = './input/'

    spellchecker = SpellChecker()
    embeddings = WordSimilarity(spellchecker)

    resultFile1 = open("result/result_top1.txt", "w")
    resultFile2 = open("result/result_top5.txt", "w")
    resultFile3 = open("result/result_visual.txt", "w")
    resultFile4 = open("result/result_text.txt", "w")
    start_time = time.time()
    for classDir in os.listdir(inputFolder1):
        predicted1 = []
        predicted5 = []
        vis_predict = []
        text_predict = []
        inputFolder = inputFolder1 + classDir + '/'
        for filename in os.listdir(inputFolder):
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                try:
                    sceneRecognizer = SceneRecognizer()
                    txtDetector = SceneTextDetector()
                    txtRecognizer = SceneTextRecognizer()
                    print(classDir + '->' + filename)
                    print('\n')

                    imgPath = rescaleImage(inputFolder, filename)


                    inds, output_prob, labels, sub_labels = sceneRecognizer.recognize(imgPath)
                    subClass = []
                    for sbl in sub_labels:
                        subClass.append(str(sbl.split(' ')[0]))

                    print "Getting visual features..."
                    visulaScores = {}
                    totalVisualScore = 0
                    for iterating_var in inds:
                        className = labels[iterating_var].split(' ')[0]
                        if className in subClass:
                            score = float(output_prob[iterating_var])
                            visulaScores[className] = score
                            totalVisualScore = score + totalVisualScore

                    for tmp in subClass:
                        tempScore = float(visulaScores[tmp]) / float(totalVisualScore)
                        visulaScores[tmp] = tempScore

                    print "Getting textual features..."
                    # scene text recognition phase
                    outputName = txtDetector.detect(imgPath)

                    # scene text recognition phase
                    words = txtRecognizer.recognize(outputName)

                    textualScores = embeddings.checkSemanticSimilarity(subClass, words)

                    print "fusing scores..."
                    finalScore = LBF(subClass, visulaScores, textualScores, 0.4, 0.6)
                    finalScore = sorted(finalScore.items(), key = operator.itemgetter(1), reverse=True)
                    finalScore = finalScore[0:5]

                    actual = subClass.index(classDir)
                    for item in finalScore:
                        index = subClass.index(item[0])
                        if finalScore.index(item) == 0:
                            value = str(index) + '|' + str(item[1])
                            predicted1.append(value)
                        if index == actual:
                            predicted5.append(actual)
                            break
                        elif finalScore.index(item) == 4:
                            predicted5.append(subClass.index(finalScore[0][0]))

                    visulaScores = sorted(visulaScores.items(), key=operator.itemgetter(1), reverse=True)
                    textualScores = sorted(textualScores.items(), key=operator.itemgetter(1), reverse=True)
                    visulaScores = visulaScores[0:5]
                    textualScores = textualScores[0:5]

                    for item in visulaScores:
                        index = subClass.index(item[0])
                        if visulaScores.index(item) == 0:
                            value = str(index) + '|' + str(item[1])
                            vis_predict.append(value)
                            break

                    for item in textualScores:
                        index = subClass.index(item[0])
                        if textualScores.index(item) == 0:
                            value = str(index) + '|' + str(item[1])
                            text_predict.append(value)
                            break




                except Exception:
                    print(colored('############ Classifying ' + str(filename) + ' has thrown error due to' + str(Exception.message), 'green'))
                    print('\n')

        resultFile1.write(str(subClass.index(classDir)) + ':' + toString(predicted1) + '\n')
        resultFile2.write(str(subClass.index(classDir)) + ':' + toString(predicted5) + '\n')
        resultFile3.write(str(subClass.index(classDir)) + ':' + toString(vis_predict) + '\n')
        resultFile4.write(str(subClass.index(classDir)) + ':' + toString(text_predict) + '\n')

    resultFile1.close()
    resultFile2.close()
    resultFile3.close()
    resultFile4.close()

    print(colored('############ Testing in % seconds ################' % (time.time() - start_time), 'green'))


def rescaleImage(imagePath,filename):
    finalImage = imagePath + filename
    img = cv2.imread(finalImage)
    imgWidth = img.shape[1]
    imgHeight = img.shape[0]
    if(imgWidth > 1500):
        r = img.shape[0] / float(imgWidth)
        dim = (1500, int(1500 * r))
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(finalImage, resized)
    elif(imgHeight > 1500):
        r = float(img.shape[1]) / imgHeight
        dim = (int(1500 * r),1500)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(finalImage, resized)
    return finalImage

def rescaleImage2(imagePath,filename):
    finalImage = imagePath + filename
    img = cv2.imread(finalImage)
    dim = (350, 350)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(finalImage, resized)
    return finalImage


#Linear Bimodal Fusion
def LBF(subclass, class_visual_prob, class_textual_prob, visual_weight, textual_weight):
    result = {}
    if(len(class_visual_prob) == len(class_textual_prob)):
        for className in subclass:
            score = (visual_weight * class_visual_prob[className]) + (textual_weight * class_textual_prob[className])
            result[className] = score
    else:
        raise Exception('inconsistency in class probabilities...')
    return result


def toString(list):
    result = " ".join(str(x) for x in list)
    return result


if __name__ == '__main__':
    main()