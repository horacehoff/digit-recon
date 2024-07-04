import os
import hashlib

training_data = []
testing_data = []
numbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
numbers_n = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
training_hashes = set()
testing_hashes = set()
for dir_name in numbers:
    for filename in os.listdir("data/training/"+dir_name+"/"):
        path = os.path.join("data/training/"+dir_name+"/", filename)
        digest = hashlib.sha1(open(path, 'rb').read()).digest()
        if digest not in training_hashes:
            training_hashes.add(digest)
            training_data.append((path, numbers_n[numbers.index(dir_name)]))
        else:
            os.remove(path)
    for filename in os.listdir("data/testing/"+dir_name+"/"):
        path = os.path.join("data/testing/"+dir_name+"/", filename)
        digest = hashlib.sha1(open(path, 'rb').read()).digest()
        if digest not in testing_hashes:
            testing_hashes.add(digest)
            testing_data.append((path, numbers_n[numbers.index(dir_name)]))
        else:
            os.remove(path)