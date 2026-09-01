TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
QT += core widgets xml
TARGET = Anguli

INCLUDEPATH += ../Headers ../Headers/Noise /usr/local/include

LIBS += -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui

HEADERS += ../Headers/dialog.h ../Headers/threading.h

SOURCES += main.cpp threading.cpp dialog.cpp generation.cpp orientmap.cpp \
           densitymap.cpp noise.cpp filter_image.cpp g_cap.cpp \
           ahaque_complex.cpp cv_type_converter.cpp \
           Noise/n_noise.cpp Noise/n_cv_type_converter.cpp

QMAKE_LFLAGS += -Wl,-rpath,/usr/local/lib
