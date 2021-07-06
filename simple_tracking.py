import torch
import torch.backends.cudnn as cudnn
import cv2
from yolact import Yolact
from utils.functions import MovingAverage
from multiprocessing.pool import ThreadPool
from queue import Queue

class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """
    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])




if __name__ == '__main__':
    vid = cv2.VideoCapture(0)
    
    if not vid.isOpened():
        print('Could not open video.')
        exit(-1)

    net = CustomDataParallel(net).cuda()
    transform = torch.nn.DataParallel(FastBaseTransform()).cuda()
    frame_times = MovingAverage(400)
    fps = 0
    # The 0.8 is to account for the overhead of time.sleep
    frame_time_target = 1 / vid.get(cv2.CAP_PROP_FPS)
    running = True

    frame_id = 0
    every_k_frames = 5
    moving_statistics = {"conf_hist": []}

    def cleanup_and_exit():
        print()
        pool.terminate()
        vid.release()
        cv2.destroyAllWindows()
        exit()

    def get_next_frame(vid):
        return [vid.read()[1] for _ in range(args.video_multiframe)]

    def transform_frame(frames):
        with torch.no_grad():
            frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
            return frames, transform(torch.stack(frames, 0))

    def eval_network(inp):
        with torch.no_grad():
            frames, imgs = inp
            if frame_idx % every_k_frames == 0 or cfg.flow.warp_mode == 'none':
                extras = {"backbone": "full", "interrupt": False, "keep_statistics": True,
                        "moving_statistics": moving_statistics}

                with torch.no_grad():
                    net_outs = net(imgs, extras=extras)

                moving_statistics["feats"] = net_outs["feats"]
                moving_statistics["lateral"] = net_outs["lateral"]

            else:
                extras = {"backbone": "partial", "interrupt": False, "keep_statistics": False,
                        "moving_statistics": moving_statistics}

                with torch.no_grad():
                    net_outs = net(imgs, extras=extras)
            frame_idx += 1

            return frames, net_outs["pred_outs"]

    def prep_frame(inp):
        with torch.no_grad():
            frame, preds = inp
            return prep_display(preds, frame, None, None, undo_transform=False, class_color=True)

    frame_buffer = Queue()
    video_fps = 0

    extract_frame = lambda x, i: (x[0][i] if x[1][i] is None else x[0][i].to(x[1][i]['box'].device), [x[1][i]])

    # Prime the network on the first frame because I do some thread unsafe things otherwise
    print('Initializing model... ', end='')
    eval_network(transform_frame(get_next_frame(vid)))
    print('Done.')

    # For each frame the sequence of functions it needs to go through to be processed (in reversed order)
    sequence = [prep_frame, eval_network, transform_frame]
    n_threads = len(sequence) + args.video_multiframe + 2
    n_threads = 4
    pool = ThreadPool(processes=n_threads)
    print("Number of threads: {}".format(n_threads))
    pool.apply_async(play_video)

    active_frames = []
    inference_times = []

    print()

    while vid.isOpened() and running:
        start_time = time.time()

        # Start loading the next frames from the disk
        next_frames = pool.apply_async(get_next_frame, args=(vid,))

        # For each frame in our active processing queue, dispatch a job
        # for that frame using the current function in the sequence
        for frame in active_frames:
            frame['value'] = pool.apply_async(sequence[frame['idx']], args=(frame['value'],))

        # For each frame whose job was the last in the sequence (i.e. for all final outputs)
        for frame in active_frames:
            if frame['idx'] == 0:
                frame_buffer.put(frame['value'].get())

        # Remove the finished frames from the processing queue
        active_frames = [x for x in active_frames if x['idx'] > 0]

        # Finish evaluating every frame in the processing queue and advanced their position in the sequence
        for frame in list(reversed(active_frames)):
            frame['value'] = frame['value'].get()
            frame['idx'] -= 1

            if frame['idx'] == 0:
                # Split this up into individual threads for prep_frame since it doesn't support batch size
                active_frames += [{'value': extract_frame(frame['value'], i), 'idx': 0} for i in range(1, args.video_multiframe)]
                frame['value'] = extract_frame(frame['value'], 0)


        # Finish loading in the next frames and add them to the processing queue
        active_frames.append({'value': next_frames.get(), 'idx': len(sequence)-1})

        # Compute FPS
        inference_time = time.time() - start_time
        frame_times.add(inference_time)
        inference_times.append(inference_time)
        fps = args.video_multiframe / frame_times.get_avg()
        np.save(args.video, np.asarray(inference_times))

        print('\rProcessing FPS: %.2f | Video Playback FPS: %.2f | Frames in Buffer: %d    ' % (fps, video_fps, frame_buffer.qsize()), end='')

    cleanup_and_exit()
