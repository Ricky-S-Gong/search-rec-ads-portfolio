export type PlanetVisual = {
  name: 'Mercury' | 'Venus' | 'Earth' | 'Mars' | 'Jupiter' | 'Saturn' | 'Uranus' | 'Neptune' | 'Pluto';
  src: string;
  width: number;
  height: number;
  x: number;
  y: number;
  size: number;
  opacity: number;
  depth: 1 | 2 | 3;
  prominence: 'distant' | 'supporting' | 'primary';
  mobile: boolean;
};

export const planets: PlanetVisual[] = [
  { name: 'Jupiter', src: '/images/cosmos/planets/jupiter-cutout.webp', width: 1254, height: 1254, x: 111, y: 56, size: 1120, opacity: .38, depth: 1, prominence: 'primary', mobile: false },
  { name: 'Mercury', src: '/images/cosmos/planets/mercury-cutout.webp', width: 420, height: 420, x: 55, y: 76, size: 44, opacity: .34, depth: 1, prominence: 'distant', mobile: false },
  { name: 'Venus', src: '/images/cosmos/planets/venus-complete-cutout-v2.webp', width: 1254, height: 1254, x: 66, y: 18, size: 92, opacity: .48, depth: 1, prominence: 'distant', mobile: false },
  { name: 'Uranus', src: '/images/cosmos/planets/uranus-cutout.webp', width: 420, height: 420, x: 48, y: 32, size: 48, opacity: .27, depth: 1, prominence: 'distant', mobile: false },
  { name: 'Neptune', src: '/images/cosmos/planets/neptune-cutout.webp', width: 480, height: 480, x: 72, y: 83, size: 68, opacity: .46, depth: 2, prominence: 'distant', mobile: false },
  { name: 'Pluto', src: '/images/cosmos/planets/pluto-cutout.webp', width: 420, height: 420, x: 42, y: 63, size: 32, opacity: .28, depth: 1, prominence: 'distant', mobile: false },
  { name: 'Mars', src: '/images/cosmos/planets/mars-cutout.webp', width: 1289, height: 1221, x: 62, y: 29, size: 142, opacity: .78, depth: 2, prominence: 'supporting', mobile: true },
  { name: 'Saturn', src: '/images/cosmos/planets/saturn-cutout.webp', width: 1672, height: 941, x: 91, y: 6, size: 610, opacity: .78, depth: 2, prominence: 'primary', mobile: true },
  { name: 'Earth', src: '/images/cosmos/planets/earth-cutout.webp', width: 1672, height: 941, x: 78, y: 55, size: 620, opacity: .96, depth: 3, prominence: 'primary', mobile: true },
];
