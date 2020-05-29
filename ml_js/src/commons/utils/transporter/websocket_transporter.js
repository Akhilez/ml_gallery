import Transporter from './transporter';
import {MLPyHost, MLPyPort} from "../../settings";

export default class WebsocketTransporter extends Transporter {
    constructor(project_id, call_back, trace_id) {
        super(project_id, call_back, trace_id);

        this.mlPyUrl = `ws://${MLPyHost}:${MLPyPort}/ws/${project_id}`;
        this.socket = new WebSocket(this.mlPyUrl);
    }

    init() {
        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.received(data);
        };

        this.socket.onclose = () => {
            console.error('Chat socket closed unexpectedly')
        };
    }

    send(data) {
        if (data.action === 'listen') return;
        data['trace_id'] = this.trace_id;
        console.log(data);
        this.socket.send(JSON.stringify(data));
    }

}