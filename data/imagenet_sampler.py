# ImageNet Data Sampler
# Rex Wang
import urllib
import os
from time import time
from tqdm import tqdm
# Pillow required
from PIL import Image
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import freeze_support
import random


class Sampler2(object):
    def __init__(self, url_path, save_image_path, multi_thread, threadPoolvolume=0):
        self.url_path = url_path
        self.save_image_path = save_image_path
        self.urls = []
        self.multi_thread = multi_thread
        self.threadPoolvolume = threadPoolvolume
        
        if not os.path.exists(save_image_path):
            os.makedirs(save_image_path)

        self.samples = 250000
        self.imagenet_fall11_size = 14197122
        # random index samples
        self.indicies = sorted(list(set(random.sample(xrange(self.imagenet_fall11_size), self.samples))))

    def url_loader(self):
        index = 0
        stop_point = self.indicies[len(self.indicies)-1]

        for line in open(self.url_path):
            if index == self.indicies[0]:
                self.indicies = self.indicies[1:]
                self.urls.append((line.split()[1], index))
            
            if index == stop_point:
                break
            index += 1

    def download_handler(self, url_tuple, extension='.jpg'):
        # filename = url_tuple[0].rsplit('/', 1)[-1]
        # filename = filename.replace('\n', '')
        filenum = str(url_tuple[1])
        try:
            urllib.urlretrieve(url=url_tuple[0], filename=self.save_image_path + filenum + extension)
            img = Image.open(self.save_image_path + filenum + extension)
            #new_img = img.resize((128, 128))
            new_img.save(self.save_image_path + filenum + extension)
        except:
            with open(self.save_image_path + str(url_tuple[1]) + '_FAIL', 'w') as failed_image:
                failed_image.write('')

    def url_downloader(self):
        if self.multi_thread:
            pool = ThreadPool(self.threadPoolvolume)
            for _ in tqdm(pool.imap_unordered(self.download_handler, self.urls), total=len(self.urls)):
                pass
        else:
            for url in tqdm(self.urls):
                self.download_handler(url)

if __name__ == '__main__':
    # Usage1: single thread version
    # Usage2: multi thread version
    imagenet_f = 'data/fall11_urls.txt'
    cat_sampler = Sampler2(url_path=imagenet_f,
                           save_image_path='data/images/imagenet_200k_samples',
                           multi_thread=True,
                           threadPoolvolume=4)
    cat_sampler.url_loader()
    print 'images: ' + str(len(cat_sampler.urls))
    cat_sampler.url_downloader()