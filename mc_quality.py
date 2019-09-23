import caiman as cm
import numpy as np
import matplotlib.pyplot as plt


data_dir = '/home/przemek/neurodata/cheeseboard-down/down_2/2019-08/habituation/2019-08-29/trial/mv_caimg/E-BL/Session1/H10_M42_S35'
vid_files = ['msCam9.avi']
mmap_files = ['msCam9_rig__d1_240_d2_376_d3_1_order_F_frames_1000_.mmap']
fnames = [data_dir + '/' + vid for vid in vid_files]
m_orig = cm.load_movie_chain(fnames)
fnames_rig = [data_dir + '/' + f for f in mmap_files]
m_rig = cm.load(fnames_rig)
bord_px_els = 10

# Do piecewise motion_correct
max_shifts = (6, 6)  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
strides =  (32, 32)  # create a new patch every x pixels for pw-rigid correction
overlaps = (16, 16)  # overlap between pathes (size of patch strides+overlaps)
num_frames_split = 100  # length in frames of each chunk of the movie (to be processed in parallel)
max_deviation_rigid = 3   # maximum deviation allowed for patch with respect to rigid shifts
pw_rigid = False  # flag for performing rigid or piecewise rigid motion correction
shifts_opencv = True  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
mc = cm.motion_correction.MotionCorrect(fnames, dview=dview, max_shifts=max_shifts,
                  strides=strides, overlaps=overlaps,
                  max_deviation_rigid=max_deviation_rigid,
                  shifts_opencv=shifts_opencv, nonneg_movie=True,
                  border_nan=border_nan)
mc.pw_rigid = True  # turn the flag to True for pw-rigid motion correction
print('Running piecewise MC')
mc.motion_correct(save_movie=True, template=None)
m_els = cm.load(mc.fname_tot_els)
cm.stop_server(dview=dview)

print('Calculating metrics')
final_size = np.subtract(m_orig.shape, 2 * bord_px_els) # remove pixels in the boundaries
winsize = 100
swap_dim = False
resize_fact_flow = .02    # downsample for computing ROF

tmpl_rig, correlations_orig, flows_orig, norms_orig, crispness_orig = cm.motion_correction.compute_metrics_motion_correction(
    fnames[0], final_size[0], final_size[1], swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)

tmpl_rig, correlations_rig, flows_rig, norms_rig, crispness_rig = cm.motion_correction.compute_metrics_motion_correction(
    fnames_rig[0], final_size[0], final_size[1],
    swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)

tmpl_els, correlations_els, flows_els, norms_els, crispness_els = cm.motion_correction.compute_metrics_motion_correction(
    mc.fname_tot_els[0], final_size[0], final_size[1],
    swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)

# Plot correlation with mean
plt.figure(figsize = (20,10))
plt.subplot(211); plt.plot(correlations_orig); plt.plot(correlations_rig); #plt.plot(correlations_els)
plt.legend(['Original','Rigid'])
plt.subplot(223); plt.scatter(correlations_orig, correlations_rig); plt.xlabel('Original');
plt.ylabel('Rigid'); plt.plot([0.3,0.7],[0.3,0.7],'r--')
axes = plt.gca(); axes.set_xlim([0.3,0.7]); axes.set_ylim([0.3,0.7]); plt.axis('square');
plt.subplot(224); plt.scatter(correlations_rig, correlations_els); plt.xlabel('Rigid');
plt.ylabel('PW-Rigid'); plt.plot([0.3,0.7],[0.3,0.7],'r--')
axes = plt.gca(); axes.set_xlim([0.3,0.7]); axes.set_ylim([0.3,0.7]); plt.axis('square');

# print crispness values
print('Crispness original: '+ str(int(crispness_orig)))
print('Crispness rigid: '+ str(int(crispness_rig)))
print('Crispness elastic: '+ str(int(crispness_els)))

# %% plot the results of Residual Optical Flow
fls = [fnames[0][:-4] + '_metrics.npz', fnames_rig[0][:-4] + '_metrics.npz', mc.fname_tot_els[0][:-4] + '_metrics.npz']
movies = [m_orig, m_rig, m_els]

plt.figure(figsize=(20, 10))
for cnt, fl, metr, movie in zip(range(len(fls)), fls, ['raw', 'rigid', 'pw-rigid'], movies):
    with np.load(fl) as ld:
        print(ld.keys())
        print(fl)
        print(str(np.mean(ld['norms'])) + '+/-' + str(np.std(ld['norms'])) +
              ' ; ' + str(ld['smoothness']) + ' ; ' + str(ld['smoothness_corr']))

        plt.subplot(len(fls), 3, 1 + 3 * cnt)
        plt.ylabel(metr)
        try:
            mean_img = np.mean(movie, 0)[12:-12, 12:-12]
        except:
            try:
                mean_img = np.mean(movie, 0)[12:-12, 12:-12]
            except:
                mean_img = np.mean(
                    cm.load(fl[:-12] + 'hdf5'), 0)[12:-12, 12:-12]

        lq, hq = np.nanpercentile(mean_img, [.5, 99.5])
        plt.imshow(mean_img, vmin=lq, vmax=hq)
        plt.title('Mean')
        plt.subplot(len(fls), 3, 3 * cnt + 2)
        plt.imshow(ld['img_corr'], vmin=0, vmax=.35)
        plt.title('Corr image')
        plt.subplot(len(fls), 3, 3 * cnt + 3)
        # plt.plot(ld['norms'])
        # plt.xlabel('frame')
        # plt.ylabel('norm opt flow')
        # plt.subplot(len(fls), 3, 3 * cnt + 3)
        flows = ld['flows']
        mean_flow_img = np.mean(np.sqrt(flows[:, :, :, 0] ** 2 + flows[:, :, :, 1] ** 2), 0)
        plt.imshow(mean_flow_img, vmin=0, vmax=0.1)
        plt.colorbar()
        plt.title('Mean optical flow')

C = cm.concatenate([m_orig, m_rig, m_els], axis=2)
#C.play(fr=60)