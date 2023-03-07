import { Audio } from 'react-loader-spinner'
type Props = {}

const Spinner = (props: Props) => {
    return (
        <Audio
            height="80"
            width="80"
            color="green"
            ariaLabel="loading"
        />
    )
}

export default Spinner