// Execute only the audited numerical fish/bar update from the pinned reference.
// No React imports, DOM events, assets, network access or upstream callbacks run.
import fs from 'node:fs';
const request = JSON.parse(fs.readFileSync(0, 'utf8'));
const source = fs.readFileSync(request.source, 'utf8');
const start = source.indexOf('        if (Math.random() <');
const end = source.indexOf('        let treasureInBar = false;');
if (start < 0 || end <= start) throw new Error('Reference source structure changed');
const body = source.slice(start, end);
const step = new Function('s', 'randRange', 'Math', `
  let {fishPos, fishTargetPos, fishSpeed, ypos, barSpeed, difficulty, motionType, length, bobber, mouseDown} = s;
  let fishAcceleration;
  ${body}
  return {fishPos, fishTargetPos, fishSpeed, ypos, barSpeed};
`);
const output = request.cases.map(test => {
  let n = 0;
  const referenceMath = Object.create(Math);
  referenceMath.random = () => {
    if (n >= test.tape.length) throw new Error('Random tape exhausted');
    return test.tape[n++];
  };
  const range = (low, high) => Math.floor(referenceMath.random() * (high-low)) + low;
  let state = {...test.state};
  return test.actions.map(held => {
    state.mouseDown = {current: !!held};
    state = {...state, ...step(state, range, referenceMath)};
    return [state.fishPos, state.fishTargetPos, state.fishSpeed, state.ypos, state.barSpeed];
  });
});
process.stdout.write(JSON.stringify(output));
