from pyquery import PyQuery as pq
import urllib
from lxml import html, etree
import re
import requests
import json
import os
import time


headers = {'Connection': 'keep-alive', 'content-type': 'text', 'X-Requested-With': 'XMLHttpRequest', 'Request-Id': '|xtzOI.Fqub3',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}


def collect(url):
    r = requests.get(url, headers=headers).text
    tree = html.fromstring(r)
    return tree


def test(tree, test_part, test_path):
    test_thing = tree.xpath(test_path)
    print(test_part + ': ', test_thing)


def down_pic(pic_urls, working_dir):
    for i, pic_url in enumerate(pic_urls):
        try:
            pic = requests.get(pic_url, timeout=15)
            saving_dir = os.path.join(working_dir, str(i + 1) + '.jpg')
            with open(saving_dir, 'wb') as f:
                f.write(pic.content)
                print('sucessfully download picture %s: %s' %
                      (str(i + 1), str(pic_url)))
        except Exception as e:
            print('fail to download picture %s: %s' %
                  (str(i + 1), str(pic_url)))
            print(e)
            continue

def save_data(info, info_type, name):
    #checking length of input information are all equal
    length = len(info[0])
    check = True
    for i in range(len(info)):
        check = (len(info[i])==length) and check

    #combine crawling infomation of a single review together
    if check:
        for i in range(length):
            review_info = {}
            for j in range(len(info)):
                review_info[info_type[j]] = info[j][i]
            with open(name + ".json", "a") as f:
                json.dump(review_info,f)
                f.write('\n')
                f.close()
                #print("saved")


def get_pageinfo(url_sample, brand_name):
    #Do the crawling
    tree = collect(url_sample)
    review_score = tree.xpath('.//article/div/div/div/span[@class="rating-value"]/text()')
    timestamp = tree.xpath('.//article/div/div/div[@class="date"]/p/text()')

    product_info = tree.xpath('.//article/div/div/div[@class="review-title"]/h4/a/text()')
    product_type = []
    brand = []
    product = []
    for i in range(int(len(product_info)/3)):
        product_type.append(product_info[int(i*3)])
        brand.append(product_info[int(i*3+1)])
        product.append(product_info[int(i*3+2)])

    review_text = tree.xpath('.//article/div/div/div[@class="__ReviewTextReadMoreV2__"]/@data-text')
    review_text_filter = []
    for i in range(int(len(review_text)/2)):
        review_text_filter.append(review_text[int(i*2)])
    #print(review_score[1])
    save_data([review_score, timestamp, product_type, brand, product, review_text_filter], ["review_score", "timestamp", "product_type", "brand", "product", "review_text"], brand_name)
    return timestamp

def get_review(url, brand_name):
    if os.path.isfile(brand_name + ".json"): 
        os.remove(brand_name + ".json")
    i = 1
    timestamp = [1]
    while len(timestamp) > 0:
        url_sample = url + "?page=" + str(i)
        if i%100 == 0:
            print(url_sample)
            print(len(timestamp))
            time.sleep(5)
        timestamp = get_pageinfo(url_sample, brand_name)
        
        i += 1
    print(i, timestamp)

def get_product_url(url_sample, brand_name):
    #Do the crawling
    tree = collect(url_sample)
    selectors = tree.xpath('.//article[@class="item product-result-row"]')
    product_url_dic = {}
    for selector in selectors:
        brand = selector.xpath('.//div[@class="sub-header"]/a[1]/text()')
        product = selector.xpath('.//div[@class="sub-header"]/a[2]/text()')
        product_url = selector.xpath('.//a[@class="item-reviews-number"]/@href')
        if len(product_url) != 0:
            if len(brand) == 0: brand = [None]
            if len(product) == 0: product = [None]
            product_url_dic[product_url[0]] = [brand[0], product[0]]
    return product_url_dic

def check_notempty(input_list):
    if len(input_list) != 0:
        return input_list[0]
    else:
        return None

def get_product_pageinfo(url_sample, brand_name, product_type):
    #Do the crawling
    #print(url_sample, "#############################")
    tree = collect(url_sample)
    selectors = tree.xpath('.//article[@class="small-image-review"]')
    check_repeat = []
    for selector in selectors:
        review_info = {}
        review_score = selector.xpath('.//div/span[@class="rating-value"]/text()')
        review_info["review_score"] = check_notempty(review_score)
        timestamp = selector.xpath('.//div[@class="date"]/p/text()')
        review_info["timestamp"] = check_notempty(timestamp)
        review_text = selector.xpath('.//div[@class="__ReviewTextReadMoreV2__"]/@data-text')

        review_text = check_notempty(review_text)
        review_info["review_text"] = review_text
        review_info["brand"] = product_type[0]
        review_info["product"] = product_type[1]
        #print(review_text)
        if review_text not in check_repeat:
            check_repeat.append(review_text)
            with open(brand_name + ".json", "a") as f:
                json.dump(review_info,f)
                f.write('\n')
                f.close()
    #save_data([review_score, timestamp, product_type, brand, product, review_text_filter], ["review_score", "timestamp", "product_type", "brand", "product", "review_text"], brand_name)
    #print([review_score, timestamp, review_text], [len(review_score), len(timestamp), len(review_text)])
    #print(len(check_repeat))
    return check_repeat

# Due to current error: 
# all reviews after 100 page return 404, 
# we change to crawl reviews through products
def get_product_review(url, brand_name):
    if os.path.isfile(brand_name + ".json"): 
        os.remove(brand_name + ".json")
    # i is the number of page
    i = 1
    info_num = 1
    product_url_all = {}
    while info_num > 0: #while info_num < 3:
        url_sample = url + "?page=" + str(i)
        if i%5 == 0:
            print("crawling product url", i)
            time.sleep(2)
        product_url_dic = get_product_url(url_sample, brand_name)
        info_num = len(product_url_dic.keys())
        product_url_all.update(product_url_dic)
        #print(url_sample, product_url_dic, info_num)
        i += 1

    for url in product_url_all:
        review_page = 1
        review_info_count = 10
        product_type = product_url_all[url]
        while review_info_count > 9:
            url_sample = url + "?page=" + str(review_page) + "#reviews"
            #print(url_sample)
            review_info_count = len(get_product_pageinfo(url_sample, brand_name, product_type))
            #print(review_info)

            if review_page%10 == 0:
                print("crawling product reviews", url_sample, review_page, review_info_count)
                time.sleep(1)
            review_page += 1

if __name__ == "__main__":
    with open("chosen_brand.txt","r") as f:
        brand_list = f.readlines()
        brand_name_list = []
        brand_score_list = []
        url_list = []
        for i in range(int((len(brand_list)+1)/4)):
            brand_name_list.append(brand_list[i*4])
            brand_score_list.append(brand_list[i*4+1])
            url_list.append(brand_list[i*4+2])
        print(brand_name_list)
        print(brand_score_list)
        print(url_list)
        print(len(url_list))
        for i in range(len(url_list)):
            print("++++++++++++++++++++++", brand_name_list[i].strip(),"+++++++++++++++++++")
            #get_review(url_list[i].strip() + "reviews", brand_name_list[i].strip())
            get_product_review(url_list[i].strip() + "products", brand_name_list[i].strip())
