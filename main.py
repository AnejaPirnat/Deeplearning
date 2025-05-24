import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import models, layers, utils, callbacks, optimizers
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("GPU naprave:", tf.config.list_physical_devices('GPU'))

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0003
IMAGE_SIZE = (64, 64)

def nalozi(mapa):
    image_paths = []
    labels = []
    #Pridobi seznam vseh podmap v glavni mapi
    vse_mape = sorted(os.listdir(mapa))

    #Gre skozi vsako podmapo
    for mapa_razreda in vse_mape:
        #Ustvari polno pot do trenutne podmape
        pot_razreda = os.path.join(mapa, mapa_razreda)
        oznaka = int(mapa_razreda)

        #gre skozi vse datoteke v trenutni podmapi
        for datoteka in os.listdir(pot_razreda):
            if datoteka.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm')):
                #doda celotno pot do slike v seznam poti
                image_paths.append(os.path.join(pot_razreda, datoteka))
                labels.append(oznaka)
    return image_paths, labels

def augmentacija(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    return img

def generator(image_paths, labels, batch_size, num_classes, augment=False, shuffle=True):
    #ustvari seznam indeksov slik (0, 1, 2, ..., n-1)
    ids = np.arange(len(image_paths))

    #generator dela v neskončni zanki, da lahko model dobi podatke večkrat (več epoh)
    while True:
        #če je shuffle=True, premešaj indekse, da bodo slike v naključnem vrstnem redu
        if shuffle:
            np.random.shuffle(ids)
        #po korakih po batch_size vzemi koščke indeksov
        for i in range(0, len(image_paths), batch_size):

            #izbere trenutno skupino indeksov za batch
            batch_idxs = ids[i:i + batch_size]
            batch_images = []  #sem bo shranilo slike za trenutni batch
            batch_labels = []   #sem bom shranilo oznake za trenutni batch

            #za vsak indeks iz batcha naloži in predela sliko ter oznako
            for j in batch_idxs:
                #naloži sliko z določeno velikostjo (npr. 64x64)
                img = load_img(image_paths[j], target_size=IMAGE_SIZE)
                #pretvori sliko v številčno matriko (array) in normaliziraj vrednosti v [0,1]
                img = img_to_array(img) / 255.0
                if augment:
                    #uporabi naključne spremembe na sliki (flip, svetlost, kontrast)
                    img = augmentacija(tf.convert_to_tensor(img))
                    #poskrbi, da so vrednosti ostale med 0 in 1 in pretvori nazaj v numpy array
                    img = tf.clip_by_value(img, 0.0, 1.0).numpy()
                #doda predelano sliko v batch
                batch_images.append(img)
                #dodaj ustrezno oznako v batch
                batch_labels.append(labels[j])
            batch_labels_onehot = utils.to_categorical(batch_labels, num_classes=num_classes)
            yield np.array(batch_images), batch_labels_onehot

def ustvari_model(velikost_slike, st_razredov, kernel_size=3):
    #ustvari zaporedni (Sequential) model
    model = models.Sequential()
    #doda vhodno plast z določeno velikostjo slike (višina, širina, 3 barvni kanali)
    model.add(layers.Input(shape=(velikost_slike[0], velikost_slike[1], 3)))

    #začni s 32 konvolucijskimi filtri
    kanali = 32
    for _ in range(3):
        #prva konvolucijska plast z funkcijo SELU
        model.add(layers.Conv2D(kanali, (kernel_size, kernel_size), padding='same', activation='selu'))
        #druga konvolucijska plast v bloku
        model.add(layers.Conv2D(kanali, (kernel_size, kernel_size), padding='same', activation='selu'))
        #max pooling plast za zmanjšanje dimenzij
        model.add(layers.MaxPooling2D((2, 2)))
        #podvoji število kanalov za naslednji blok
        kanali *= 2

    #uporabi globalno povprečenje, da zmanjša dimenzije pred gosto plastjo
    model.add(layers.GlobalAveragePooling2D())
    #gosta plast s 128 nevroni in SELU
    model.add(layers.Dense(128, activation='selu'))
    model.add(layers.AlphaDropout(0.3))
    model.add(layers.Dense(st_razredov, activation='softmax'))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def treniraj_in_prikazi(mapa_slik, velikost_slike, kernel_sizes, epochs=EPOCHS, batch_size=BATCH_SIZE):
    image_paths, labels = nalozi(mapa_slik)
    st_razredov = len(np.unique(labels))

    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    history_dict = {}

    if not os.path.exists("models"):
        os.makedirs("models")

    for ks in kernel_sizes:
        print(f"\nTreniranje modela s kernel velikostjo {ks}x{ks}")
        model = ustvari_model(velikost_slike, st_razredov, kernel_size=ks)

        train_gen = generator(X_train, y_train, batch_size, st_razredov, augment=True)
        val_gen = generator(X_val, y_val, batch_size, st_razredov, augment=False, shuffle=False)

        history = model.fit(
            train_gen,
            steps_per_epoch=len(X_train) // batch_size,
            validation_data=val_gen,
            validation_steps=len(X_val) // batch_size,
            epochs=epochs,
            callbacks=[
                callbacks.EarlyStopping(patience=7, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6),
                TqdmCallback(verbose=1)
            ],
            verbose=0
        )

        history_dict[ks] = history

        ime_modela = f"model_kernel_{ks}x{ks}_epoh_{epochs}.keras"
        model.save(os.path.join("models", ime_modela))

    # Graf validacijske izgube
    plt.figure(figsize=(12, 5))
    for ks, history in history_dict.items():
        plt.plot(history.history['val_loss'], label=f'Val izguba {ks}x{ks}')
    plt.title("Primerjava validacijske izgube")
    plt.xlabel("Epoh")
    plt.ylabel("Izguba")
    plt.legend()
    plt.show()

    # Graf validacijske točnosti
    plt.figure(figsize=(12, 5))
    for ks, history in history_dict.items():
        plt.plot(history.history['val_accuracy'], label=f'Val točnost {ks}x{ks}')
    plt.title("Primerjava validacijske točnosti")
    plt.xlabel("Epoh")
    plt.ylabel("Točnost")
    plt.legend()
    plt.show()

    for ks, history in history_dict.items():
        print(f"Končna validacijska točnost za kernel {ks}x{ks}: {history.history['val_accuracy'][-1]:.4f}")

import matplotlib.pyplot as plt


def glavni_program():
    mapa_slik = "GTSRB/Training"  # prilagodi pot do tvojih slik
    kernel_sizes = [3, 5, 7]
    treniraj_in_prikazi(mapa_slik, IMAGE_SIZE, kernel_sizes)


if __name__ == "__main__":
    glavni_program()


