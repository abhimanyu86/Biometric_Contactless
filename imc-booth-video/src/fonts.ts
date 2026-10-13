import { loadFont as loadOswald } from '@remotion/google-fonts/Oswald';
import { loadFont as loadInter } from '@remotion/google-fonts/Inter';

export const oswald = loadOswald('normal', { weights: ['500', '600', '700'], subsets: ['latin'] }).fontFamily;
export const inter = loadInter('normal', { weights: ['400', '500', '600', '700'], subsets: ['latin'] }).fontFamily;
