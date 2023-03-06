import axios from "axios";
import { AudioExtension } from "./types/recorder";
const ext: AudioExtension = 'wav'

export const saveAudio = async (blob: Blob, uuid: string) => {
    let formData = new FormData();
    let fileName = `${uuid}.${ext}`;
    let file = new File([blob], fileName);
    formData.append('file', file, fileName);
    try {
        const response = await axios.post(`${import.meta.env.VITE_API}/audio`,
            formData, {
            headers: {
                'Content-Type': `multipart/form-data`,
            },
        })
        console.log(response);
    } catch (err) {
        console.log(err);
    }
}

export const deleteAudio = async (uuid: string) => {
    try {
        const response = await axios.delete(`${import.meta.env.VITE_API}/audio/${uuid}`)
        console.log(response);
    } catch (err) {
        console.log(err);
    }
}

export const predictAudio = async (uuid: string) => {
    try {
        const response = await axios.get(`${import.meta.env.VITE_API}/audio/predict${uuid}`)
        console.log(response);
        return response.data
    } catch (err) {
        console.log(err);
    }
}

export const getAudio = async (uuid: string) => {
    try {
        const response = await axios.get(`${import.meta.env.VITE_API}/audio/${uuid}.${ext}`)
        console.log(response);
        return response.data
    } catch (err) {
        console.log(err);
    }
}

export async function getAudios() {
    try {
        const response = await axios.get(`${import.meta.env.VITE_API}/audio`)
        console.log(response);
        return response.data
    } catch (err) {
        console.log(err);
    }
}