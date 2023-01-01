# Data

Preprocessed data included in [data/dBTM/] could be used directly for dBTM, O-dBTM. It can also be used for baseline BTM, dJST and TBIP, with some tiny change to fit the input formats of those models.
The original data is from [MakeupAlley](https://www.makeupalley.com/), a review website on beauty products.

## Data info
### In the `data/{dataset_name}/clean`, you should find the following files inside this folder:

- `brand_map.txt`: a `[num_brands]`-length file where each line denotes the name of the brand in the corpus.
- `product_map.txt`: a `[num_products]`-length file where each line denotes the product type in the corpus.
- `vocabulary.txt`: a `[num_words]`-length file where each line denotes the corresponding word in the vocabulary.

### In the `data/{dataset_name}/time`, you should find the following files inside this folder:

- `counts.npz`: a `[num_documents, num_words]`
  [sparse CSR matrix](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.html)
  containing the word counts for each document.
- `brand_indices.npy`: a `[num_documents]` vector where each entry is an integer in the set `{0, 1, ..., num_brands - 1}`, indicating the brand of the corresponding document in `counts.npz`.
- `products.npy`: a `[num_documents]` vector where each entry is an integer in the set `{0, 1, ..., num_products - 1}`, indicating the product type of the corresponding document in `counts.npz`.
- `score_indices.npy`: a `[num_documents]` vector where each entry is an integer in the set `{-1, 0, 1}`, indicating the review polarity of the corresponding document in `counts.npz`.
- `score_indices_initial.npy`: a `[num_documents]` vector where each entry is an integer in the set `{1, 2, 3, 4, 5}`, indicating the original review rating of the corresponding document in `counts.npz`.
- `raw_documents.txt`: including review text. This is removed from the submitted version due to limitation of the data size.

## Extra info
- The timestamps of reviews are indentified by the folder name they are in: `{0, 1, ..., 8}` is actually the year `{2005, 2006, ..., 2013}`, you can find more statistics information about this data in the Appendix of dBTM paper.
- 25 brands are in this dataset, including `Dior`, `Estée Lauder`, `E.L.F. Cosmetics`, etc.
- 111 product type are in this dataset, including `Body Lotion & Cream`, `Makeup Brushes`, `Shampoo & Conditioner`, etc.

## Example review
- I love the idea of a tinted lip balm : a slight sheen of color with all the down-to-earth benefit and ease of use of a lip balm ! The bright packaging is cute and fun , and the lip balm itself smell like grape-flavored candy . The color of the lip balm itself look intimidatingly dark purple but go on a sheer , soft berry . I like it well enough , especially since it includes SPF . However , I do notice that my lip start feeling dry more quickly than when I use my plain old Chapstick , so I may experiment with other brand of tinted lip balm before I try other color of this line .
- I wa really excited for these during it initial release since they were all over Youtube and sadly , I fell for the hype . I chose Pink Possibilities and Who wore it red-er for my NC30 skintone . Description from Maybelline 's website , `` Pure color pigment suspended in a weightless gel . No heavy wax or oil . Soft , sexy gel-color . '' Even though the red is more opaque than the pink , the formula is still too sheer for my liking . These last around two hour and definitely not through a meal . I need to wear a lip balm ( blistex in the blue tube ) under or else my lip get flakey , it 's not moisturizing enough on it own . I paid \\$ 8 CAD at Walmart , for 0.11oz / 3g . These were in my abandoned makeup drawer , just took them out today so I can try to use it a few more time . But I will not repurchase . The only thing I like is the sleek packaging . Overall , I prefer Revlon 's lip butter ( usually on sale for $ 4 at Shoppers Drug Mart / Superstore . )
- I just love this foundation so much ! I have it in the shade 845 warm beige . It set to my skin really fast and feel like a light powder.It finish matte on my skin and it feel completely weightless ! My sister hate the feeling of makeup on her face and she hate the tiniest bit of makeup on her face but she say it feel like shes not wearing anything ! It really last and i dont get an oily feeling . I wore it for about 7hrs and it still perfect ! The color seemed a tiny bit light when i first applied it but about 10 min later it just match my skin perfectly ! The only thing is that i wont recommend it for people with dry skin because it doe show the dry patch quite a bit , but otherwise it is my HG foundation !


