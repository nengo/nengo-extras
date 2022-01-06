from io import BytesIO
from time import time as unix_time

import nengo
import numpy as np
import PIL.Image
import PIL.ImageFile

from nengo_extras import reqs

if reqs.HAS_GI:
    import gi
    from gi.repository import Gst


class CameraPipeline:
    def __init__(self, device="/dev/video0", width=640, height=480):
        if not reqs.HAS_GI:
            raise ImportError("`CameraPipeline` requires `pygobject`")

        gi.require_version("Gst", "1.0")

        self.Gst = Gst

        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.channels = 3
        self.size = self.width * self.height * self.channels
        assert self.size > 0

        self.running = False
        self.pipeline = None
        self.source = None
        self.parse = None
        self.sink = None

        self.time = -1
        self.jpeg = None

        self.init_gst()

    def init_gst(self):
        self.Gst.init(None)

        self.pipeline = self.Gst.Pipeline.new("camera")

        self.source = self.Gst.ElementFactory.make("v4l2src", "source")
        self.source.set_property("device", self.device)

        self.parse = self.Gst.ElementFactory.make("jpegparse", "jpeg-parser")

        self.sink = self.Gst.ElementFactory.make("appsink", "sink")
        self.sink.set_property("drop", True)
        self.sink.set_property("max-buffers", 3)
        self.sink.set_property("emit-signals", True)
        self.sink.set_property("sync", False)
        self.sink.connect("new-sample", self.frame_from_sink)

        self.pipeline.add(self.source)
        self.pipeline.add(self.parse)
        self.pipeline.add(self.sink)

        caps = self.Gst.caps_from_string(
            "image/jpeg,width=%d,height=%d,framerate=30/1" % (self.width, self.height)
        )
        self.source.link_filtered(self.parse, caps)
        self.parse.link(self.sink)

    def frame_from_sink(self, sink):
        sample = sink.emit("pull-sample")
        if sample is not None:
            buffer = sample.get_buffer()
            string = buffer.extract_dup(0, buffer.get_size())
            self.time = unix_time()
            self.jpeg = string

        return False

    def start(self):
        if not self.running:
            self.running = True
            self.pipeline.set_state(self.Gst.State.PAUSED)
            self.pipeline.set_state(self.Gst.State.PLAYING)

    def stop(self):
        self.pipeline.set_state(self.Gst.State.NULL)


class CameraData:
    def __init__(self, width, height, channels=3, keep_aspect=True):
        PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

        self.width = width
        self.height = height
        self.channels = channels
        self.keep_aspect = keep_aspect

        self.time = -1
        self.jpeg = None
        self.array = None

    def set_jpeg(self, time, jpeg):
        self.time = time
        self.jpeg = jpeg

        buffer = BytesIO()
        buffer.write(jpeg)
        buffer.seek(0)
        image = PIL.Image.open(buffer)
        camshape = np.array([image.height, image.width])

        if self.keep_aspect:
            aspect = float(self.width) / self.height

            crop0 = min(camshape[0], camshape[1] / aspect)
            cropshape = np.array([int(crop0), int(crop0 * aspect)])
            crop_a = (camshape - cropshape) // 2
            crop_b = crop_a + cropshape
            image = image.crop((crop_a[1], crop_a[0], crop_b[1], crop_b[0]))

        image = image.resize((self.width, self.height))
        array = np.asarray(image)
        array = np.transpose(array, (2, 0, 1))  # put color channel first
        self.array = array


class Camera(nengo.Process):
    def __init__(
        self,
        width=256,
        height=256,
        offset=-128,
        device="/dev/video0",
        cam_width=640,
        cam_height=480,
    ):
        self.width = int(width)
        self.height = int(height)
        self.offset = offset
        self.device = device
        self.cam_width = int(cam_width)
        self.cam_height = int(cam_height)

        self.framerate = 0.1

        # Assume 3 channels
        super().__init__(
            default_size_in=0, default_size_out=self.width * self.height * 3
        )

    def make_step(self, shape_in, shape_out, dt, rng, state):
        assert shape_in == (0,)
        pipeline = CameraPipeline(
            device=self.device, width=self.cam_width, height=self.cam_height
        )
        channels = pipeline.channels
        size = self.width * self.height * channels
        assert size > 0
        assert shape_out == (size,)

        data = CameraData(self.width, self.height, pipeline.channels)
        zero_image = np.zeros((pipeline.channels, self.width, self.height))
        offset = self.offset
        framerate = self.framerate

        def step_camera(t):
            # TODO: stop the pipeline when the simulation ends
            pipeline.start()

            if pipeline.time - data.time > framerate:
                data.set_jpeg(pipeline.time, pipeline.jpeg)

            array = data.array if data.array is not None else zero_image
            array = array + offset
            return array.ravel()

        return step_camera
