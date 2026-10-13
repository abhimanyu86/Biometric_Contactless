// Brand palette — sampled from the YellowSense poster (see assets/brand.json).
export const C = {
  cream: '#FBF7F0',
  paper: '#FFFFFF',
  ink: '#15120E',
  mute: '#6B6358',
  orange: '#C26E10',
  orangeDeep: '#A85A0A',
  orangeSoft: '#F8E7CE',
  mustard: '#C98A1C',
  line: '#E8DECF',
  green: '#2E8B57',
  greenSoft: '#E3F3E9',
  dark: '#17130F',
};

export const W = 1920;
export const H = 1080;
export const FPS = 30;

// Poster pillars that the demo footage does not actually show. Off by default so the loop only
// claims what is demonstrated; flip to true to add them to the "pillars" scene.
export const SHOW_UNDEMONSTRATED_POSTER_CLAIMS = false;

// The poster's "Explore more" QR decodes to a Google Drive folder link. It is reproduced because it is on the
// printed poster — confirm that folder is public and intended for visitors, or set this to false.
export const SHOW_QR = true;

// The 85% figure is the PoC acceptance gate (">85% user-wise, contactless-to-contactless"). The repo has no measurement
// that backs it yet, so it is shown as a TARGET. Only set true (and edit narration B12 in make_narration.py) once a
// measured, documented result exists.
export const ACCURACY_IS_MEASURED = false;

export const BOOTH = {
  event: 'India Mobile Congress',
  eventShort: 'IMC',
  hall: 'Hall 1',
  booth: 'TS 14',
  web: 'www.yellowsense.in',
  email: 'prakhar@yellowsense.in',
  phone: '+91 98693 97868',
  tagline: 'Contactless Identity Verification for the AI Era',
  company: 'YellowSense Technologies Pvt. Ltd.',
};
