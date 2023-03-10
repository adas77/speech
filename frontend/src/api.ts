import axios from "axios"
import { AudioExtension } from "./types/recorder"
import { v4 as uuidv4 } from 'uuid'

const ext: AudioExtension = 'wav'


export const saveAudio = async (file: File | Blob) => {
    let formData = new FormData()
    let uuid = uuidv4()
    let fileName = `${uuid}.${ext}`
    let file_
    if (file instanceof Blob) {
        file_ = new File([file], fileName)
    }
    else { file_ = file }
    formData.append('file', file_, fileName)
    try {
        const response = await axios.post(`${import.meta.env.VITE_API}/audio`,
            formData, {
            headers: {
                'Content-Type': `multipart/form-data`,
            },
        })
        console.log(response)
    } catch (err) {
        console.log(err)
    }
}


export const deleteAudio = async (uuid: string) => {
    try {
        const response = await axios.delete(`${import.meta.env.VITE_API}/audio/${uuid}`)
        console.log(response)
        return response
    } catch (err) {
        console.log(err)
    }
}

// export const predictAudio = async (uuid: string) => {
//     try {
//         const response = await axios.get(`${import.meta.env.VITE_API}/audio/predict/${uuid}`)
//         console.log(response)
//         return response.data
//     } catch (err) {
//         console.log(err)
//     }
// }

export async function getAudios() {
    try {
        const response = await axios.get(`${import.meta.env.VITE_API}/audio`)
        console.log(response)
        return response.data
    } catch (err) {
        console.log(err)
    }
}