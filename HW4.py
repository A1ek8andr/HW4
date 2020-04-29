
import numpy
import tools
import matplotlib.pyplot as plt
import numpy.fft as fft


if __name__ == '__main__':
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * numpy.pi

    # Шаг по пространству
    dx = 0.5e-3

    eps1 = 1.1
    d1 = 0.03
    samplebegin1 = int(0.15 / dx)
    sampleend1 = samplebegin1 + int(d1 / dx)

    eps2 = 2.2
    d2 = 0.04
    samplebegin2 = sampleend1
    sampleend2 = samplebegin2 + int(d2 / dx)

    eps3 = 4
    d3 = 0.06
    samplebegin3 = sampleend2
    sampleend3 = samplebegin3 + int(d3 / dx)

    eps4 = 4.5
    samplebegin4 = sampleend3

    # Размер области моделирования
    x = 0.35

    # Число Куранта
    Sc = 1.0

    # Скорость света
    c = 3e8

    # Шаг по времени
    dt = Sc * dx / c
    print(f"Шаг временной сетки: {dt} с")

    # Время расчета в отсчетах
    maxTime = 4000

    # Размер области моделирования в отсчетах
    maxSize = int(x / dx)

    # Положение источника в отсчетах
    sourcePos = 50

    # Датчики для регистрации поля
    probesPos = [25, 80]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    eps = numpy.ones(maxSize)
    eps[samplebegin1:] = eps1
    eps[samplebegin2:] = eps2
    eps[samplebegin3:] = eps3
    eps[samplebegin4:] = eps4
    Ez = numpy.zeros(maxSize)
    Hy = numpy.zeros(maxSize - 1)

    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.1
    display_ymax = 1.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel, dx)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])
    display.drawBoundary(samplebegin1)
    display.drawBoundary(samplebegin2)
    display.drawBoundary(samplebegin3)
    display.drawBoundary(samplebegin4)

    # Параметры гауссова импульса
    A0 = 100  # ослабление в 0 момент времени по отношение к максимуму
    Am = 100  # ослабление на частоте Fm
    Fm = 10e9
    wg = numpy.sqrt(numpy.log(Am)) / (numpy.pi * Fm)
    NWg = wg / dt
    dg = wg * numpy.sqrt(numpy.log(A0))
    NDg = dg / dt

    for t in range(maxTime):
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / W0

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= (Sc / W0) * \
            numpy.exp(-((t - NDg - sourcePos) / NWg) ** 2)
        # Hy[sourcePos - 1] -= (Sc / W0) * numpy.exp(-((t - NDg)/NWg) ** 2 )

        # Граничные условия для поля E
        Ez[0] = Ez[1]
        oldboundary = Ez[-2]

        # Расчет компоненты поля E
        Hy_shift = Hy[:-1]
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy_shift) * Sc * W0 / eps[1: -1]
        Ez[-1] = oldboundary + (Sc - numpy.sqrt(eps[-1])) / \
            (Sc + numpy.sqrt(eps[-1])) * (Ez[-2] - Ez[-1])
        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += (Sc / (numpy.sqrt(eps[sourcePos]))) * \
            numpy.exp(-(((t + 0.5) - (sourcePos - 0.5) - NDg) / NWg) ** 2)
        # Ez[sourcePos] += Sc * numpy.exp(-(((t + 0.5) - (-0.5) - NDg)/NWg) ** 2 )

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if t % 10 == 0:
            display.updateData(display_field, t)

    display.stop()

    # Отображение сигнала, сохраненного в датчиках
    tools.showProbeSignals(probes, -1.1, 1.1, dt)

    # Размера БПФ
    size = 4096*2

    # Выдедение падающего поля из датчика на 80 отсчете
    FallField = numpy.zeros(maxTime)
    FallField[0:400] = probes[1].E[0:400]

    # Нахождение БПФ падающего поля
    FallSpectr = abs(fft.fft(FallField, size))
    FallSpectr = fft.fftshift(FallSpectr)

    # Нахождение БПФ отраженного поля
    ScatteredSpectr =abs(fft.fft(probes[0].E, size))
    ScatteredSpectr = fft.fftshift(ScatteredSpectr)

    # шаг по частоте и определение частотной оси
    df = 1 / (size * dt)
    f = numpy.arange(-(size / 2) * df, (size / 2) * df, df)

    # Построение спектра падающего и рассеянного поля
    plt.figure()
    plt.plot(f * 1e-9, FallSpectr / numpy.max(FallSpectr))
    plt.plot(f * 1e-9, ScatteredSpectr / numpy.max(ScatteredSpectr))
    plt.grid()
    plt.xlim(0, 10e9 * 1e-9)
    plt.xlabel('f, ГГц')
    plt.ylabel('|S/Smax|')
    plt.legend(['Спектр падающего поля', 'Спектр отраженного поля'])

    # Определение коэффициента отражения и построения графика
    plt.figure()
    plt.plot(f * 1e-9, (ScatteredSpectr / FallSpectr))
    plt.xlim(0, 10e9 * 1e-9)
    plt.ylim(0, 0.4)
    plt.grid()
    plt.xlabel('f, ГГц')
    plt.ylabel('|Г|')
    plt.show()
