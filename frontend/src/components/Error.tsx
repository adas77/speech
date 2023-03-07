import React from 'react'

type Props = {
    msg: string
}

const Error = (props: Props) => {
    return (
        <>
            <div>Error</div>
            <p>{props.msg}</p>
        </>
    )
}

export default Error