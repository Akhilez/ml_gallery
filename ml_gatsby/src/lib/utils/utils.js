export function isCursorInScope(p5, height, width) {
  return (
    p5.mouseX > 0 && p5.mouseX < width && p5.mouseY > 0 && p5.mouseY < height
  )
}
