const NeuralGraph = require("./NeuralGraph")
// @ponicode
describe("setup", () => {
    let inst

    beforeEach(() => {
        inst = new NeuralGraph.default()
    })

    test("0", () => {
        let callFunction = () => {
            inst.setup({ DEGREES: 6370000 }, "Maurice Purdy")
        }
    
        expect(callFunction).not.toThrow()
    })

    test("1", () => {
        let callFunction = () => {
            inst.setup([520, 70], "Ronald Keeling")
        }
    
        expect(callFunction).not.toThrow()
    })

    test("2", () => {
        let callFunction = () => {
            inst.setup({ DEGREES: 2000.0 }, "Ronald Keeling")
        }
    
        expect(callFunction).not.toThrow()
    })

    test("3", () => {
        let callFunction = () => {
            inst.setup([520, 410], "Gail Hoppe")
        }
    
        expect(callFunction).not.toThrow()
    })

    test("4", () => {
        let callFunction = () => {
            inst.setup({ DEGREES: 100000 }, "Ronald Keeling")
        }
    
        expect(callFunction).not.toThrow()
    })

    test("5", () => {
        let callFunction = () => {
            inst.setup(undefined, undefined)
        }
    
        expect(callFunction).not.toThrow()
    })
})

// @ponicode
describe("preload", () => {
    let inst

    beforeEach(() => {
        inst = new NeuralGraph.default()
    })

    test("0", () => {
        let callFunction = () => {
            inst.preload(90)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("1", () => {
        let callFunction = () => {
            inst.preload(350)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("2", () => {
        let callFunction = () => {
            inst.preload(1)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("3", () => {
        let callFunction = () => {
            inst.preload(100)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("4", () => {
        let callFunction = () => {
            inst.preload(50)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("5", () => {
        let callFunction = () => {
            inst.preload(Infinity)
        }
    
        expect(callFunction).not.toThrow()
    })
})

// @ponicode
describe("drawUpdateNeuronsButtons", () => {
    let inst

    beforeEach(() => {
        inst = new NeuralGraph.default()
    })

    test("0", () => {
        let callFunction = () => {
            inst.drawUpdateNeuronsButtons(-100, 4, "bar")
        }
    
        expect(callFunction).not.toThrow()
    })

    test("1", () => {
        let callFunction = () => {
            inst.drawUpdateNeuronsButtons(-100, 410, -1)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("2", () => {
        let callFunction = () => {
            inst.drawUpdateNeuronsButtons(-100, 0.0, 10)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("3", () => {
        let callFunction = () => {
            inst.drawUpdateNeuronsButtons(-100, "bar", 1)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("4", () => {
        let callFunction = () => {
            inst.drawUpdateNeuronsButtons(1, 50, 1)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("5", () => {
        let callFunction = () => {
            inst.drawUpdateNeuronsButtons(NaN, NaN, undefined)
        }
    
        expect(callFunction).not.toThrow()
    })
})

// @ponicode
describe("drawFlower2", () => {
    let inst

    beforeEach(() => {
        inst = new NeuralGraph.default()
    })

    test("0", () => {
        let callFunction = () => {
            inst.drawFlower2({ petalWidth: "Extensions", petalHeight: "Extensions", sepalWidth: "Jean-Philippe", sepalHeight: true })
        }
    
        expect(callFunction).not.toThrow()
    })

    test("1", () => {
        let callFunction = () => {
            inst.drawFlower2({ petalWidth: "Extensions", petalHeight: "Expressway", sepalWidth: "Pierre Edouard", sepalHeight: false })
        }
    
        expect(callFunction).not.toThrow()
    })

    test("2", () => {
        let callFunction = () => {
            inst.drawFlower2({ petalWidth: "Port", petalHeight: "Extensions", sepalWidth: "Pierre Edouard", sepalHeight: true })
        }
    
        expect(callFunction).not.toThrow()
    })

    test("3", () => {
        let callFunction = () => {
            inst.drawFlower2({ petalWidth: "Extensions", petalHeight: "Extensions", sepalWidth: "Michael", sepalHeight: true })
        }
    
        expect(callFunction).not.toThrow()
    })

    test("4", () => {
        let callFunction = () => {
            inst.drawFlower2({ petalWidth: "Expressway", petalHeight: "Port", sepalWidth: "Anas", sepalHeight: true })
        }
    
        expect(callFunction).not.toThrow()
    })

    test("5", () => {
        let callFunction = () => {
            inst.drawFlower2({ petalWidth: undefined, petalHeight: undefined, sepalWidth: undefined, sepalHeight: undefined })
        }
    
        expect(callFunction).not.toThrow()
    })
})

// @ponicode
describe("drawSepal", () => {
    let inst

    beforeEach(() => {
        inst = new NeuralGraph.default()
    })

    test("0", () => {
        let callFunction = () => {
            inst.drawSepal(1, 50, 1.5, 576)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("1", () => {
        let callFunction = () => {
            inst.drawSepal(70, 1, 150, 400)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("2", () => {
        let callFunction = () => {
            inst.drawSepal(90, 30, 9, 2)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("3", () => {
        let callFunction = () => {
            inst.drawSepal(1, 400, 48, 24000)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("4", () => {
        let callFunction = () => {
            inst.drawSepal(550, 410, 48000, 99)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("5", () => {
        let callFunction = () => {
            inst.drawSepal(-Infinity, undefined, undefined, undefined)
        }
    
        expect(callFunction).not.toThrow()
    })
})

// @ponicode
describe("drawPetal", () => {
    let inst

    beforeEach(() => {
        inst = new NeuralGraph.default()
    })

    test("0", () => {
        let callFunction = () => {
            inst.drawPetal(520, 410, 390, 48)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("1", () => {
        let callFunction = () => {
            inst.drawPetal(520, 90, 100, 320)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("2", () => {
        let callFunction = () => {
            inst.drawPetal(320, 1, 0.0, 432)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("3", () => {
        let callFunction = () => {
            inst.drawPetal(520, 1, 0, 0.5)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("4", () => {
        let callFunction = () => {
            inst.drawPetal(4, 520, 150, 680)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("5", () => {
        let callFunction = () => {
            inst.drawPetal(-Infinity, -Infinity, -Infinity, -Infinity)
        }
    
        expect(callFunction).not.toThrow()
    })
})

// @ponicode
describe("drawFlowerCenter", () => {
    let inst

    beforeEach(() => {
        inst = new NeuralGraph.default()
    })

    test("0", () => {
        let callFunction = () => {
            inst.drawFlowerCenter(30, 70, 320, 400)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("1", () => {
        let callFunction = () => {
            inst.drawFlowerCenter(50, 520, 720, 40)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("2", () => {
        let callFunction = () => {
            inst.drawFlowerCenter(410, 90, 150, 480)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("3", () => {
        let callFunction = () => {
            inst.drawFlowerCenter(70, 400, 12, 40)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("4", () => {
        let callFunction = () => {
            inst.drawFlowerCenter(320, 90, 2, 24000)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("5", () => {
        let callFunction = () => {
            inst.drawFlowerCenter(undefined, undefined, undefined, undefined)
        }
    
        expect(callFunction).not.toThrow()
    })
})

// @ponicode
describe("drawFlower", () => {
    let inst

    beforeEach(() => {
        inst = new NeuralGraph.default()
    })

    test("0", () => {
        let callFunction = () => {
            inst.drawFlower()
        }
    
        expect(callFunction).not.toThrow()
    })
})

// @ponicode
describe("drawClassificationBox", () => {
    let inst

    beforeEach(() => {
        inst = new NeuralGraph.default()
    })

    test("0", () => {
        let callFunction = () => {
            inst.drawClassificationBox(50)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("1", () => {
        let callFunction = () => {
            inst.drawClassificationBox(410)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("2", () => {
        let callFunction = () => {
            inst.drawClassificationBox(4)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("3", () => {
        let callFunction = () => {
            inst.drawClassificationBox(320)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("4", () => {
        let callFunction = () => {
            inst.drawClassificationBox(70)
        }
    
        expect(callFunction).not.toThrow()
    })

    test("5", () => {
        let callFunction = () => {
            inst.drawClassificationBox(-Infinity)
        }
    
        expect(callFunction).not.toThrow()
    })
})

// @ponicode
describe("mouseClicked", () => {
    let inst

    beforeEach(() => {
        inst = new NeuralGraph.default()
    })

    test("0", () => {
        let callFunction = () => {
            inst.mouseClicked({ mouseX: 0, x: 320, mouseY: 10, y: 90, w: 50, h: -5.48 })
        }
    
        expect(callFunction).not.toThrow()
    })

    test("1", () => {
        let callFunction = () => {
            inst.mouseClicked({ mouseX: 400, x: 90, mouseY: 0, y: 400, w: 100, h: 100 })
        }
    
        expect(callFunction).not.toThrow()
    })

    test("2", () => {
        let callFunction = () => {
            inst.mouseClicked({ mouseX: 90, x: 30, mouseY: 0, y: 400, w: 520, h: 0 })
        }
    
        expect(callFunction).not.toThrow()
    })

    test("3", () => {
        let callFunction = () => {
            inst.mouseClicked({ mouseX: 400, x: 320, mouseY: -10, y: 100, w: 50, h: 0 })
        }
    
        expect(callFunction).not.toThrow()
    })

    test("4", () => {
        let callFunction = () => {
            inst.mouseClicked({ mouseX: 30, x: 90, mouseY: 0.0, y: 550, w: 550, h: 1 })
        }
    
        expect(callFunction).not.toThrow()
    })

    test("5", () => {
        let callFunction = () => {
            inst.mouseClicked({ mouseX: Infinity, x: Infinity, mouseY: Infinity, y: Infinity, w: Infinity, h: Infinity })
        }
    
        expect(callFunction).not.toThrow()
    })
})
