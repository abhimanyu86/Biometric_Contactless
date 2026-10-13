import { Config } from '@remotion/cli/config';

// PNG frames + BT.709 + yuv420p => limited-range HD video that TVs / USB players / signage boxes handle correctly.
Config.setVideoImageFormat('png');
Config.setOverwriteOutput(true);
Config.setCodec('h264');
Config.setCrf(17);
Config.setPixelFormat('yuv420p');
Config.setColorSpace('bt709');
