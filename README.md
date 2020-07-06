 # HSVM

![logo](logo/logo.png)

HSVM - программа для обучения SVM классификатора с использованием HOG дескриптора на OpenCV и C++. 
## Библиотеки для сборки

- OpenCV версии 3.0.0 и выше

## [Example](/example)
* [train SVM](main.cpp)

### Подготовка изображений для обучения

SVM классификатор используют для решения бинарных задач, то есть когда ответ классификатора может быть да или нет. Например надо классифицировать собаку, тогда нам понадобиться фотографии с собакой (на всеx фотографиях должна быть собака, без лишнего фона), и фотографии фона, то есть фотографии, где нет собаки (фотографии фона могут быть любыми, на них может быть любой объект, кроме классифицируемого). Фотографии собаки и фотографии фона, должна находиться в разных папках. Если вам необходимо классифицировать более одного объекта, то надо обучить столько классификаторов, сколько классов вам нужно классифицировать.

### Сборка проекта 

Сборка на Windows

!ВАЖНО! Перед сборкой проекта в файле [CMakeLists.txt](example/load_model/CMakeLists.txt), на строке set(OpenCV_DIR "C:/Program Files/opencv/build") под номером 28, укажите путь до вашей папки с собранной библиотекой OpenCV.
```cmd
git clone https://github.com/Artemy2807/hsvm.git hsvm
cd hsvm
mkdir build && cd build
cmake -A x64 ..
MSBuild.exe train-svm.sln -property:Configuration=Debug
cd Debug
```

Сборка на Linux
```bash
git clone https://github.com/Artemy2807/hsvm.git hsvm
cd hsvm
mkdir build && cd build
cmake ..
make -j4
```

### Запуск программы

После сборки проекта, запускаем программу для обучения SVM. Аргументы программы:
- -g - путь до папки с фотографиями объекта, который вы хотите классифицировать.
- -b - путь до папки с фотографиями фона.
- -s - размер входных данных для обучения. Первым передаётся ширина изображения, вторым передаётся высота изображения. Размер указывается через запятую без пробела. Если ваши фотографии для обучения имеют другой размер, то программа сама изменит иx размер. Размер должен быть больше или равно 16 пикселям. Также размер должен соотвестовать степенью двойки.
Windows
```cmd
train-svm -g img/good -b img/bad -s 64,64 -o output.xml
```

Linux
```bash
./train-svm -g img/good -b img/bad -s 64,64 -o output.xml
```

После завершения работы программы появится файл output.xml, это файл обученного SVM классификатора. Ниже показано как его использовать.

### Использование SVM

Начало программы
```c++
#include <opencv2/opencv.hpp>
#include <vector>

int main() {
```

Загружаем SVM классификатор/ы, для дальнейшего использования. Размер вектора SVMDetectors должен быть, такого размера, сколько классификаторов вы хотите загрузить. Далее загружаем SVM классификаторы. Если вы хотите загрузить более одного классификатора, то в каждый элемент вектора загружаем классификатор.

Пример загрузки одного классификатора:
```c++
	std::vector<cv::Ptr<cv::ml::SVM>> SVMDetectors{ 1 };
	SVMDetectors[0] = cv::Algorithm::load<cv::ml::SVM>("output.xml");
```
- "output.xml" - путь до обученного классификатора

Пример загрузки нескольких классификаторов:
```c++
	std::vector<cv::Ptr<cv::ml::SVM>> SVMDetectors{ 3 };
	SVMDetectors[0] = cv::Algorithm::load<cv::ml::SVM>("output_1.xml");
    SVMDetectors[1] = cv::Algorithm::load<cv::ml::SVM>("output_2.xml");
    SVMDetectors[2] = cv::Algorithm::load<cv::ml::SVM>("output_3.xml");
```
- "output_1.xml", "output_2.xml", "output_3.xml" - пути до обученных классификаторов.

Перед использованием SVM, необходимо подготовить данные для классификации. Создаём класс HOGDescriptor, для использования HOG дескриптора, после чего указываем размер данных. Выше мы обучали классификатор, тот размер, который вы указывали для обучения, необходимо вписать. Загружаем картинку и изменяем её размер. Далее используем HOG дескриптор, на загруженном изображение. descriptors - вектор, содержащий данные, извлечёнными HOG дескриптором. После чего конвертируем вектор, в формат необходимый для SVM классификатора.
```c++
	cv::HOGDescriptor hog;
    cv::Size size(64, 64);      // Сюда впишите свой размер данных
	hog.winSize = size;  
    cv::Mat img = cv::imread("img.jpg");        // Укажите путь до вашей картинки
    cv::resize(img, img, size);
	std::vector<float> descriptors;
	hog.compute(resized, descriptors, cv::Size(8, 8), cv::Size(0, 0));
    cv::Mat gradientList = cv::Mat(cv::Size(descriptors.size(), 1), CV_32FC1);
	transpose(cv::Mat(descriptors), gradientList);
```

Мы подготовили данные для классификации, теперь используем SVM классификатор/ы. В цикле проходимся по загруженным SVM классификаторам. predict - ответ SVM классификатора. Если ответ SVM классификатора ближе к 1, то на изображение класс, на который был обучен SVM классификатор, а если ответ классификатора ближе к -1, то это фон. Чтобы определить, какой SVM классификатор сработал, используйте индекс i. i - положение SVM классификатора в векторе.
```c++
    for(size_t i = 0; i < SVMDetectors.size(); i++) {
        float predict = SVMDetectors[i]->predict(gradientList);
        // Используйте predict и i для ваших целей
    }
    return 0;
}
```

## Contacts
Автор: Одышев Артемий
- Telegram: [@artemy](https://t.me/artemy_odeshev)
- VK: [@artemy](https://vk.com/artemyodiesiev)


 