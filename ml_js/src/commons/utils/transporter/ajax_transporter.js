import Transporter from "./transporter";
import {MLPyHost, MLPyPort} from "../../settings";


export default class AjaxTransporter extends Transporter {
    constructor(project_id, call_back, trace_id) {
        super(project_id, call_back, trace_id);

        this.mlPyUrl = `http://${MLPyHost}:${MLPyPort}/ajax/${project_id}`;
    }

    init() {
        this.send({action: 'init'});
    }

    send(data) {
        data['trace_id'] = this.trace_id;

        fetch(this.mlPyUrl, {
            method: "POST",
            body: JSON.stringify(data)
        }).then(res => {
            let received_data = res.json();
            this.received(received_data);
        });
    }

}
