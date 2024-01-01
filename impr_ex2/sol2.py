import numpy as np
import imageio
from skimage import color
from scipy import signal
from scipy.ndimage import map_coordinates
from scipy import io
import matplotlib.pyplot as plt


TESTING = False


def DFT(signal):
    signal_array = signal
    if len(signal.shape) == 2:
        signal_array = np.transpose(signal)[0]
    x = np.arange(signal_array.shape[0])
    u = x.reshape((signal_array.shape[0], 1))
    result = np.dot(signal_array, np.exp(-2j * np.pi * u * x / signal_array.shape[0]))
    if len(signal.shape) == 2:
        result = result.reshape(signal.shape)
    return result


def IDFT(fourier_signal):
    fourier_signal_array = fourier_signal
    if len(fourier_signal.shape) == 2:
        fourier_signal_array = np.transpose(fourier_signal)[0]
    u = np.arange(fourier_signal_array.shape[0])
    x = u.reshape((fourier_signal_array.shape[0], 1))
    result = np.dot(fourier_signal_array, np.exp(2j * np.pi * u * x / fourier_signal_array.shape[0])) / \
             fourier_signal_array.shape[0]
    if len(fourier_signal.shape) == 2:
        result = result.reshape(fourier_signal.shape)
    # for my testing:
    if TESTING:
        result = result.astype('int16')
    return result


def DFT2(image):
    ret_image = image[:].astype('complex128')
    for row_idx, row in enumerate(ret_image):
        ret_image[row_idx, :] = DFT(row)
    for col_idx in range(ret_image.shape[1]):
        ret_image[:, col_idx] = DFT(ret_image[:, col_idx])
    return ret_image


def IDFT2(fourier_image):
    ret_image = fourier_image[:].astype('complex128')
    for row_idx, row in enumerate(ret_image):
        ret_image[row_idx, :] = IDFT(row)
    for col_idx in range(ret_image.shape[1]):
        ret_image[:, col_idx] = IDFT(ret_image[:, col_idx])
    return ret_image


def change_rate(filename, ratio):
    rate, data = io.wavfile.read(filename)
    io.wavfile.write("./change_rate.wav", int(rate * ratio), data)
    return


def resize(data, ratio):
    fourier_data = DFT(data)
    fourier_data = np.fft.fftshift(fourier_data)
    n = len(fourier_data)
    new_size = int(np.floor(n / ratio))
    if ratio >= 1:
        trim_amount_before = int(round((n - new_size) / 2))
        trim_amount_after = n - new_size - trim_amount_before
        resized_fourier_data = fourier_data[trim_amount_before: - trim_amount_after]
    else:
        zeros_num_before = int(np.round((new_size - n) / 2))
        zeros_num_after = new_size - n - zeros_num_before
        resized_fourier_data = np.concatenate((np.zeros(zeros_num_before), fourier_data, np.zeros(zeros_num_after)))
    return IDFT(np.fft.ifftshift(resized_fourier_data)).astype(data.dtype)


def change_samples(filename, ratio):
    rate, data = io.wavfile.read(filename)
    resized_data = resize(data, ratio)
    io.wavfile.write("./change_samples.wav", rate, resized_data)
    return


def resize_spectrogram(data, ratio):
    sp_data = stft(data)
    resized_sp_data = np.apply_along_axis(resize, 1, sp_data, ratio=ratio)
    return istft(resized_sp_data).astype(data.dtype)


def resize_vocoder(data, ratio):
    sp_data = stft(data)
    resized_sp_data = phase_vocoder(sp_data, ratio)
    return istft(resized_sp_data).astype(data.dtype)


def conv_der(im):
    dx = signal.convolve2d(im, np.array([[0.5, 0, -0.5]]), mode='same')
    dy = signal.convolve2d(im, np.array([[0.5, 0, -0.5]]).reshape((3, 1)), mode='same')
    return np.sqrt(dx ** 2 + dy ** 2)


def fourier_der(im):
    fourier_im = DFT2(im)
    fourier_im_centered = np.fft.fftshift(fourier_im)
    fourier_dx = compute_dx_der(fourier_im_centered)
    # fourier_dy = compute_dy_der(fourier_im_centered)
    fourier_dy = np.transpose(compute_dx_der(np.transpose(fourier_im_centered)))
    dx = IDFT2(np.fft.ifftshift(fourier_dx))
    dy = IDFT2(np.fft.ifftshift(fourier_dy))
    return np.real(np.sqrt(dx ** 2 + dy ** 2)).astype('float64')
    # return np.abs(dy)


def compute_dx_der(centered_fourier_im):
    n = centered_fourier_im.shape[1]
    if n % 2 == 1:
        u = np.arange(-(n // 2), (n // 2) + 1)
    else:
        u = np.arange(-n // 2, n // 2)
    return centered_fourier_im * u * 2j * np.pi / n


##################################################################################################################


def read_image(filename, representation):
    """
    Reads an image
    :param filename: path to image file
    :param representation: controls convertion of output: 1 for grayscale output or 2 for RGB output
    :return: np.ndarray, type float64, normalized to [0,1]
    """
    image = imageio.imread(filename)
    if representation == 1:
        return color.rgb2gray(image).astype(np.float64)
    return (image / 255).astype(np.float64)


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec


if __name__ == '__main__':
    # sig = np.array([[5, 3, 6, 4, 6, 2, 1], [1, 2, 3, 4, 5, 6, 7]])
    # print(DFT2(sig))
    # print(np.isclose(IDFT2(DFT2(sig)), sig))
    # print(np.fft.fft2(sig))
    # print(np.isclose(DFT2(sig), np.fft.fft2(sig)))

    # external_path = "C:\\Users\\danie\\Desktop\\Uni\\YearD\\Image_Processing\\ex2-azulay93\\external"
    # change_rate(external_path + "\\aria_4kHz.wav", 1.5)
    # change_samples(external_path + "\\aria_4kHz.wav", 1.5)
    # rate1, data1 = io.wavfile.read(external_path + "\\gettysburg10.wav")
    # data2 = resize_spectrogram(data1, 2)
    # data3 = resize_vocoder(data1, 2)
    # io.wavfile.write("./spect_resized.wav", rate1, data2)
    # io.wavfile.write("./vocoder_resized.wav", rate1, data3)

    # z = np.zeros(81).reshape(9, 9)
    # z[4, 4] = 80
    # print(f"\nz is:\n\n{z}")
    # print('\nconvolution derivative is:\n\n', conv_der(z), "\n")
    # print('\nlog fourier derivative is:\n\n', np.round(np.log(1 + np.abs(fourier_der(z))), 5))
    image_path = "C:\\Users\\danie\\Desktop\\Uni\\YearD\\Image_Processing\\ex2-azulay93\\"
    image_name = "monkey.jpg"
    image = read_image(image_path + image_name, 1)
    der_image = conv_der(image)
    der_image2 = fourier_der(image)
    im_to_display = der_image / np.max(der_image)
    plt.imshow(im_to_display, cmap='gray')
    plt.show()
    im_to_display = der_image2 / np.max(der_image2)
    plt.imshow(im_to_display, cmap='gray')
    plt.show()



