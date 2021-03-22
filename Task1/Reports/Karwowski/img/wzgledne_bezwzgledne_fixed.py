import matplotlib.pyplot as plt

labels = [
    'dolnośląskie', 'kujawsko-pomorskie', 'lubelskie', 'lubuskie', 'łódzkie',
    'małopolskie', 'mazowieckie', 'opolskie', 'podkarpackie', 'podlaskie',
    'pomorskie', 'śląskie', 'świętokrzyskie', 'warmińsko-mazurskie',
    'wielkopolskie', 'zachodniopomorskie'
]

relative = [
    4.19, 3.4, 0.86, 4.9, 3.52, 3.54, 5.34, 2.45, 3.15, 1.42, 5.25, 4.12, 3.49,
    2.18, 3.87, 3.05
]

absolute = [
    1215, 704, 181, 495, 862, 1207, 2899, 240, 669, 167, 1231, 1859, 429, 310,
    1355, 517
]

x = range(len(labels))

plt.subplots_adjust(bottom=0.2)
plt.subplot(211)
plt.ylabel("Bezwzględna liczba zachorowań")
plt.xticks([])
plt.bar(x, absolute)

plt.subplot(212)
plt.ylabel("Liczba zachorowań na 10 tys. mieszkańców")
plt.xticks(x, labels, rotation='vertical')
plt.bar(x, relative)
plt.show()
